"""Motor de planificación de sprints — TFM, Fase 4.

Asigna las historias del backlog a sprints con dos métodos:
  - M3: programación lineal entera (PuLP + CBC), ecuaciones (1)-(5) de la memoria.
  - M2: heurística greedy por razón valor ponderado / story points (línea base).
Calcula las métricas de calidad de la Tabla 4 para cualquier plan (también para el manual, M1).

Entrada: lista de historias con claves 'clave', 'story_points', 'valor_negocio',
'prioridad_num' y 'depende_de' (formato de salida de jira_ingesta, Fase 3).
"""
import math
import time

import pandas as pd
import pulp


# --------------------------------------------------------------------------- datos
def preparar(backlog):
    """Normaliza la entrada (DataFrame o lista de dicts) y descarta dependencias externas."""
    filas = backlog.to_dict("records") if isinstance(backlog, pd.DataFrame) else list(backlog)
    incompletas = [f["clave"] for f in filas
                   if pd.isna(f.get("story_points")) or pd.isna(f.get("valor_negocio"))
                   or float(f["story_points"]) <= 0]
    if incompletas:   # sin estimación o sin valor no se pueden planificar (la Fase 3 ya las reporta)
        print(f"Aviso: se excluyen {len(incompletas)} historias sin story points o sin valor: {incompletas}")
    filas = [f for f in filas if f["clave"] not in incompletas]
    claves = {f["clave"] for f in filas}
    historias = {}
    for f in filas:
        deps = f.get("depende_de") or []
        if isinstance(deps, str):
            deps = [d.strip() for d in deps.split(",") if d.strip()]
        historias[f["clave"]] = {
            "p": float(f["story_points"]),
            "v": float(f["valor_negocio"]),
            "r": float(f.get("prioridad_num") or 0),
            "deps": [d for d in deps if d in claves],
        }
    return historias


def profundidad(historias):
    """Longitud de la cadena de dependencias más larga (nº de sprints mínimo por dependencias)."""
    nivel = {}

    def calc(h):
        if h not in nivel:
            nivel[h] = 1 + max((calc(k) for k in historias[h]["deps"]), default=0)
        return nivel[h]

    return max((calc(h) for h in historias), default=0)


def horizonte(historias, capacidad, ajustar_dependencias=False):
    """Mínimo teórico de sprints + 1 de holgura (diseño experimental).

    Con ajustar_dependencias=True no baja de la profundidad del grafo de dependencias,
    porque una historia solo puede ir en un sprint posterior al de las historias de las que depende.
    """
    total = sum(h["p"] for h in historias.values())
    n = math.ceil(total / capacidad) + 1
    return max(n, profundidad(historias)) if ajustar_dependencias else n


def pesos_sprint(n):
    """w_s = (n - s + 1) / n: decrece con cada sprint."""
    return {s: (n - s + 1) / n for s in range(1, n + 1)}


def valor_ponderado(h, lam):
    return h["v"] + lam * h["r"]


# --------------------------------------------------------------------------- M3: PLE
def _binaria(prob, nombre):
    """Variable binaria compatible con PuLP 2.x (LpVariable) y 4.x (add_variable)."""
    if hasattr(prob, "add_variable"):
        return prob.add_variable(nombre, cat="Binary")
    return pulp.LpVariable(nombre, cat="Binary")


def _solver_cbc(mensajes, limite_s, warm=False):
    """CBC: PULP_CBC_CMD en PuLP 2.x; COIN_CMD (paquete pulp[cbc]) en PuLP 3+/4.x."""
    clase = getattr(pulp, "PULP_CBC_CMD", None) or pulp.COIN_CMD
    return clase(msg=mensajes, timeLimit=limite_s, warmStart=warm, gapRel=0)   # óptimo probado


def planificar_ple(historias, capacidad, n_sprints, lam=10.0, limite_s=300, mensajes=False, arranque=True):
    S = list(range(1, n_sprints + 1))
    w = pesos_sprint(n_sprints)
    H = list(historias)
    prob = pulp.LpProblem("planificacion_sprints", pulp.LpMaximize)
    x = {h: {s: _binaria(prob, f"x_{i}_{s}") for s in S} for i, h in enumerate(H)}        # (5)
    prob += pulp.lpSum(valor_ponderado(historias[h], lam) * w[s] * x[h][s]
                       for h in H for s in S)                                              # (1)
    for h in H:
        prob += pulp.lpSum(x[h][s] for s in S) <= 1, f"unica_{h}"                          # (2)
    for s in S:
        prob += pulp.lpSum(historias[h]["p"] * x[h][s] for h in H) <= capacidad, f"cap_{s}"  # (3)
    for h in H:
        for k in historias[h]["deps"]:
            for s in S:                                                                    # (4)
                prob += (pulp.lpSum(x[h][t] for t in S if t <= s)
                         <= pulp.lpSum(x[k][t] for t in S if t < s)), f"dep_{h}_{k}_{s}"
    t0 = time.perf_counter()
    if arranque:   # solución inicial factible: la del greedy (acelera la búsqueda de CBC)
        inicial, _ = planificar_greedy(historias, capacidad, n_sprints, lam)
        for h in H:
            for s in S:
                x[h][s].setInitialValue(1 if inicial[h] == s else 0)
    resultado = prob.solve(_solver_cbc(mensajes, limite_s, warm=arranque))
    segundos = time.perf_counter() - t0
    plan = {h: next((s for s in S if (x[h][s].value() or 0) > 0.5), None) for h in H}
    return plan, {"estado": _estado(prob, resultado), "segundos": segundos,
                  "variables": len(H) * len(S), "restricciones": prob.numConstraints() if hasattr(prob, "numConstraints") else len(prob.constraints)}


def _estado(prob, resultado):
    """Texto del estado de la solución en PuLP 2.x (LpStatus) y 4.x (LpSolveStats)."""
    if hasattr(pulp, "LpStatus"):
        return pulp.LpStatus[prob.status]
    nombre = getattr(getattr(resultado, "status", None), "name", str(resultado))
    return "Optimal" if nombre == "GapLimit" else nombre   # con gapRel=0, GapLimit = óptimo probado


# --------------------------------------------------------------------------- M2: greedy
def planificar_greedy(historias, capacidad, n_sprints, lam=10.0):
    t0 = time.perf_counter()
    orden = sorted(historias, key=lambda h: (-valor_ponderado(historias[h], lam) / historias[h]["p"], h))
    carga = {s: 0.0 for s in range(1, n_sprints + 1)}
    plan = {h: None for h in historias}
    cambio = True
    while cambio:                       # se repite hasta que no sea posible asignar más historias
        cambio = False
        for h in orden:
            if plan[h] is not None:
                continue
            deps = historias[h]["deps"]
            if any(plan[k] is None for k in deps):
                continue
            desde = max((plan[k] for k in deps), default=0) + 1
            for s in range(desde, n_sprints + 1):
                if carga[s] + historias[h]["p"] <= capacidad:
                    plan[h], carga[s], cambio = s, carga[s] + historias[h]["p"], True
                    break
    return plan, {"estado": "Heuristic", "segundos": time.perf_counter() - t0}


# --------------------------------------------------------------------------- métricas (Tabla 4)
def metricas(historias, plan, capacidad, n_sprints, lam=10.0):
    w = pesos_sprint(n_sprints)
    asignadas = {h: s for h, s in plan.items() if s is not None}
    z = sum(valor_ponderado(historias[h], lam) * w[s] for h, s in asignadas.items())
    carga = {s: 0.0 for s in range(1, n_sprints + 1)}
    for h, s in asignadas.items():
        carga[s] += historias[h]["p"]
    mitad = n_sprints // 2
    valor_total = sum(h["v"] for h in historias.values())
    viol_dep = sum(1 for h, s in asignadas.items() for k in historias[h]["deps"]
                   if plan.get(k) is None or plan[k] >= s)
    return {
        "Z": round(z, 2),
        "valor_temprano_pct": round(100 * sum(historias[h]["v"] for h, s in asignadas.items() if s <= mitad)
                                    / valor_total, 1),
        "utilizacion_pct": round(100 * sum(carga.values()) / (capacidad * n_sprints), 1),
        "sprints_sobrecargados": sum(1 for c in carga.values() if c > capacidad + 1e-9),
        "dependencias_violadas": viol_dep,
        "sin_asignar": len(historias) - len(asignadas),
        "sp_sin_asignar": sum(historias[h]["p"] for h in historias if plan.get(h) is None),
    }


def plan_a_tabla(historias, plan):
    return pd.DataFrame([{"clave": h, "sprint": s, "story_points": historias[h]["p"],
                          "valor_negocio": historias[h]["v"], "prioridad_num": historias[h]["r"],
                          "depende_de": ",".join(historias[h]["deps"])}
                         for h, s in plan.items()]).sort_values(["sprint", "clave"], na_position="last")


def carga_por_sprint(historias, plan, n_sprints):
    carga = {s: 0.0 for s in range(1, n_sprints + 1)}
    for h, s in plan.items():
        if s is not None:
            carga[s] += historias[h]["p"]
    return carga


# --------------------------------------------------------------------------- comparación
def comparar(backlog, capacidad=25, lam=10.0, repeticiones=5, limite_s=300, n_sprints=None,
             ajustar_dependencias=False):
    """Ejecuta M2 y M3 'repeticiones' veces y devuelve métricas, planes e info de cómputo."""
    historias = preparar(backlog)
    n = n_sprints or horizonte(historias, capacidad, ajustar_dependencias)
    filas, planes = [], {}
    for metodo, fn in (("M2 greedy", planificar_greedy), ("M3 PLE", planificar_ple)):
        tiempos = []
        for _ in range(repeticiones):
            kw = {"limite_s": limite_s} if metodo == "M3 PLE" else {}
            plan, info = fn(historias, capacidad, n, lam, **kw)
            tiempos.append(info["segundos"])
        planes[metodo] = plan
        filas.append({"metodo": metodo, "estado": info["estado"], "sprints": n,
                      "tiempo_medio_s": round(sum(tiempos) / len(tiempos), 4),
                      **metricas(historias, plan, capacidad, n, lam)})
    tabla = pd.DataFrame(filas)
    z_opt = tabla.loc[tabla.metodo == "M3 PLE", "Z"].iloc[0]
    tabla["pct_del_optimo"] = (100 * tabla["Z"] / z_opt).round(2)
    tabla["brecha_pct"] = (100 - tabla["pct_del_optimo"]).round(2)
    return tabla, planes, historias, n


def curva_valor(historias, plan, n_sprints):
    """Porcentaje acumulado del valor de negocio total entregado al final de cada sprint."""
    total = sum(h["v"] for h in historias.values())
    acum, curva = 0.0, {}
    for s in range(1, n_sprints + 1):
        acum += sum(historias[h]["v"] for h, sp in plan.items() if sp == s)
        curva[s] = round(100 * acum / total, 1)
    return curva
