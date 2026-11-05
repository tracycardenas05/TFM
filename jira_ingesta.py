"""Módulo de ingesta del backlog desde Jira Cloud (API REST v3) — TFM, Fase 3.

Uso básico:
    from jira_ingesta import JiraCloud, extraer, procesar
    jira = JiraCloud(url, email, token)
    datos = extraer(jira, "BIB", etiqueta="escenario-E1")
    df, avisos, resumen = procesar(datos, carpeta="salidas/E1", velocidad=25)
"""
import json
import math
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests

NOMBRES_STORY_POINTS = ["Story point estimate", "Story Points",
                        "Estimación de puntos de historia", "Puntos de historia"]
NOMBRE_VALOR_NEGOCIO = "Valor de negocio"
TIPOS_HISTORIA = ("Story", "Historia")
PRIORIDAD_NUM = {"Highest": 5, "High": 4, "Medium": 3, "Low": 2, "Lowest": 1,
                 "Más alta": 5, "Alta": 4, "Media": 3, "Baja": 2, "Más baja": 1}
PREFIJO_ESCENARIO = "escenario-"


# --------------------------------------------------------------------------- cliente
class JiraCloud:
    def __init__(self, url, email, token, timeout=30):
        self.base = url.rstrip("/")
        self.timeout = timeout
        self.s = requests.Session()
        self.s.auth = (email, token)
        self.s.headers.update({"Accept": "application/json", "Content-Type": "application/json"})
        self.peticiones = 0

    def _req(self, metodo, ruta, **kw):
        for _ in range(5):
            self.peticiones += 1
            r = self.s.request(metodo, f"{self.base}{ruta}", timeout=self.timeout, **kw)
            if r.status_code == 429:                       # límite de peticiones
                time.sleep(int(r.headers.get("Retry-After", 5)))
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"{metodo} {ruta} -> {r.status_code}: {r.text[:500]}")
            return r.json() if r.content else None
        raise RuntimeError(f"{metodo} {ruta}: demasiados reintentos")

    def get(self, ruta, **kw):
        return self._req("GET", ruta, **kw)

    def post(self, ruta, json):
        return self._req("POST", ruta, json=json)

    def buscar(self, jql, campos, tam_pagina=100):
        """Búsqueda JQL con paginación por nextPageToken (/rest/api/3/search/jql)."""
        issues, token, paginas = [], None, 0
        while True:
            cuerpo = {"jql": jql, "fields": campos, "maxResults": tam_pagina}
            if token:
                cuerpo["nextPageToken"] = token
            r = self.post("/rest/api/3/search/jql", cuerpo)
            paginas += 1
            issues += r.get("issues", [])
            token = r.get("nextPageToken")
            if r.get("isLast", True) or not token:
                return issues, paginas

    def campos_proyecto(self, proyecto):
        """IDs de story points, valor de negocio y sprint del proyecto.

        En los espacios gestionados por el equipo cada proyecto tiene su propio campo
        'Valor de negocio', por eso se resuelve con la pantalla de creación del proyecto
        y, si no se puede, por nombre en el catálogo global.
        """
        sp = vn = None
        try:
            tipos = self.get(f"/rest/api/3/issue/createmeta/{proyecto}/issuetypes")
            tipos = tipos.get("issueTypes") or tipos.get("values") or []
            historia = next((t for t in tipos if t["name"] in TIPOS_HISTORIA), None)
            if historia:
                r = self.get(f"/rest/api/3/issue/createmeta/{proyecto}/issuetypes/{historia['id']}")
                campos = r.get("fields") or r.get("values") or r.get("results") or []
                por_nombre = {c["name"]: c["fieldId"] for c in campos}
                sp = next((por_nombre[n] for n in NOMBRES_STORY_POINTS if n in por_nombre), None)
                vn = por_nombre.get(NOMBRE_VALOR_NEGOCIO)
        except RuntimeError:
            pass
        globales = {}
        for f in self.get("/rest/api/3/field"):
            globales.setdefault(f["name"], []).append(f["id"])
        sp = sp or next((globales[n][0] for n in NOMBRES_STORY_POINTS if n in globales), None)
        vn = vn or (globales.get(NOMBRE_VALOR_NEGOCIO) or [None])[0]
        sprint = (globales.get("Sprint") or [None])[0]
        return {"story_points": sp, "valor_negocio": vn, "sprint": sprint}


# --------------------------------------------------------------------------- extracción
def construir_jql(proyecto, etiqueta=None, solo_pendientes=True):
    jql = f'project = "{proyecto}"'
    if etiqueta:
        jql += f' AND labels = "{etiqueta}"'
    if solo_pendientes:
        jql += " AND statusCategory != Done"
    return jql + " ORDER BY key ASC"


def extraer(jira, proyecto, etiqueta=None, solo_pendientes=True):
    """Descarga las historias y devuelve la respuesta cruda con sus metadatos."""
    t0 = time.perf_counter()
    peticiones0 = jira.peticiones
    campos = jira.campos_proyecto(proyecto)
    lista = ["summary", "issuetype", "status", "priority", "labels", "issuelinks"]
    lista += [c for c in campos.values() if c]
    jql = construir_jql(proyecto, etiqueta, solo_pendientes)
    issues, paginas = jira.buscar(jql, lista)
    return {
        "meta": {
            "proyecto": proyecto, "etiqueta": etiqueta, "jql": jql, "campos": campos,
            "sitio": jira.base, "fecha_extraccion": datetime.now(timezone.utc).isoformat(),
            "paginas": paginas, "peticiones": jira.peticiones - peticiones0,
            "segundos": round(time.perf_counter() - t0, 3),
        },
        "issues": issues,
    }


# --------------------------------------------------------------------------- normalización
def _terminada(issue_vinculada):
    cat = issue_vinculada.get("fields", {}).get("status", {}).get("statusCategory", {})
    return cat.get("key") == "done"


def _es_historia(tipo):
    if tipo.get("subtask"):
        return False
    if "hierarchyLevel" in tipo:
        return tipo["hierarchyLevel"] == 0
    return tipo.get("name", "").lower() not in ("epic", "épica")


def _numero(v):
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def normalizar(datos):
    campos = datos["meta"]["campos"]
    filas = []
    for it in datos["issues"]:
        f = it["fields"]
        if not _es_historia(f["issuetype"]):
            continue
        prioridad = (f.get("priority") or {}).get("name", "")
        etiquetas = f.get("labels", [])
        escenario = next((e[len(PREFIJO_ESCENARIO):] for e in etiquetas
                          if e.startswith(PREFIJO_ESCENARIO)), None)
        sprints = f.get(campos.get("sprint") or "", None) or []
        sprint = sprints[-1] if sprints else {}
        # En la issue X, un vínculo Blocks con 'inwardIssue' = Y significa "X is blocked by Y"
        pendientes, resueltas = [], []
        for l in f.get("issuelinks", []):
            if l["type"]["name"] == "Blocks" and "inwardIssue" in l:
                y = l["inwardIssue"]
                (resueltas if _terminada(y) else pendientes).append(y["key"])
        filas.append({
            "clave": it["key"],
            "resumen": f["summary"],
            "escenario": escenario,
            "tipo": f["issuetype"]["name"],
            "estado": f["status"]["name"],
            "categoria_estado": f["status"].get("statusCategory", {}).get("key"),
            "prioridad": prioridad,
            "prioridad_num": PRIORIDAD_NUM.get(prioridad),
            "story_points": _numero(f.get(campos.get("story_points") or "")),
            "valor_negocio": _numero(f.get(campos.get("valor_negocio") or "")),
            "etiquetas": [e for e in etiquetas if not e.startswith(PREFIJO_ESCENARIO)],
            "depende_de": sorted(set(pendientes)),
            "deps_resueltas": sorted(set(resueltas)),
            "sprint": sprint.get("name") if isinstance(sprint, dict) else None,
            "sprint_id": sprint.get("id") if isinstance(sprint, dict) else None,
            "sprint_estado": sprint.get("state") if isinstance(sprint, dict) else None,
        })
    return filas


# --------------------------------------------------------------------------- validación
def validar(filas):
    """Devuelve una lista de avisos (categoría, detalle)."""
    avisos = []
    claves = {r["clave"] for r in filas}
    for r in filas:
        if r["story_points"] is None:
            avisos.append(("sin story points", r["clave"]))
        if r["valor_negocio"] is None:
            avisos.append(("sin valor de negocio", r["clave"]))
        if r["prioridad_num"] is None:
            avisos.append(("sin prioridad", r["clave"]))
        for d in r["depende_de"]:
            if d not in claves:
                avisos.append(("dependencia fuera del backlog", f"{r['clave']} -> {d}"))
    grafo = {r["clave"]: [d for d in r["depende_de"] if d in claves] for r in filas}
    estado = {}

    def visitar(n, camino):
        estado[n] = 1
        for m in grafo[n]:
            if estado.get(m) == 1:
                avisos.append(("ciclo de dependencias", " -> ".join(camino[camino.index(m):] + [m])
                               if m in camino else f"{n} -> {m}"))
            elif m not in estado:
                visitar(m, camino + [m])
        estado[n] = 2

    for n in grafo:
        if n not in estado:
            visitar(n, [n])
    return avisos


def agrupar_avisos(avisos):
    grupos = {}
    for cat, det in avisos:
        grupos.setdefault(cat, []).append(det)
    return grupos


def resumen(nombre, filas, avisos, velocidad):
    sp = sum(r["story_points"] or 0 for r in filas)
    smin = math.ceil(sp / velocidad) if sp else 0
    grupos = agrupar_avisos(avisos)
    return {
        "escenario": nombre, "historias": len(filas), "story_points": sp,
        "valor_total": sum(r["valor_negocio"] or 0 for r in filas),
        "con_dependencias": sum(1 for r in filas if r["depende_de"]),
        "vinculos": sum(len(r["depende_de"]) for r in filas),
        "sin_sp": len(grupos.get("sin story points", [])),
        "sin_valor": len(grupos.get("sin valor de negocio", [])),
        "sin_prioridad": len(grupos.get("sin prioridad", [])),
        "deps_externas": len(grupos.get("dependencia fuera del backlog", [])),
        "ciclos": len(grupos.get("ciclo de dependencias", [])),
        "sprints_min": smin, "horizonte": smin + 1 if smin else 0,
        "valido": not avisos,
    }


# --------------------------------------------------------------------------- persistencia
def guardar(datos, filas, avisos, carpeta):
    os.makedirs(carpeta, exist_ok=True)
    with open(f"{carpeta}/respuesta_jira.json", "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=1)
    with open(f"{carpeta}/backlog.json", "w", encoding="utf-8") as f:
        json.dump(filas, f, ensure_ascii=False, indent=2)
    df = pd.DataFrame(filas)
    for col in ("etiquetas", "depende_de", "deps_resueltas"):
        df[col] = df[col].apply(lambda v: ",".join(v))
    df.to_csv(f"{carpeta}/backlog.csv", index=False, encoding="utf-8-sig")
    with open(f"{carpeta}/validacion.json", "w", encoding="utf-8") as f:
        json.dump(agrupar_avisos(avisos), f, ensure_ascii=False, indent=2)


def cargar_crudo(ruta):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def procesar(datos, carpeta=None, velocidad=25, nombre=None):
    """Normaliza, valida y (opcionalmente) guarda. Funciona igual en línea y sin conexión."""
    filas = normalizar(datos)
    avisos = validar(filas)
    if carpeta:
        guardar(datos, filas, avisos, carpeta)
    nombre = nombre or datos["meta"].get("etiqueta") or datos["meta"]["proyecto"]
    return pd.DataFrame(filas), avisos, resumen(nombre, filas, avisos, velocidad)
