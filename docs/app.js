"use strict";

const $ = (s, r = document) => r.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const nf1 = new Intl.NumberFormat("es-AR", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const nf0 = new Intl.NumberFormat("es-AR");
const M = (x) => (x == null ? "–" : `${nf1.format(x / 1e6)} M`);
const N = (x) => (x == null ? "–" : nf0.format(x));
const A = (x) => (x == null ? "–" : x.toFixed(4).replace(".", ","));
const signo = (x) => (x == null ? "–" : `${x > 0 ? "+" : x < 0 ? "−" : ""}${nf1.format(Math.abs(x) / 1e6)} M`);
const claseSigno = (x) => (x == null || x === 0 ? "" : x > 0 ? "gana" : "pierde");
const MESES_ES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"];
const nombreMes = (m) => { const t = MESES_ES[Number(String(m).slice(4)) - 1] || String(m); return t[0].toUpperCase() + t.slice(1); };
const NOMBRE_MES = new Proxy({}, { get: (_, m) => nombreMes(m) });

const ESTADO_EXP = {
  base: ["adoptado", "base"],
  confirmado: ["adoptado", "adoptado"],
  a_confirmar: ["pendiente", "a confirmar"],
  descartado: ["descartado", "descartado"],
  rechazado: ["rechazado", "no se confirmó"],
  error: ["fallido", "error"],
  fuera_de_presupuesto: ["fallido", "se pasó de tiempo"],
};
const ESTADO_IDEA = {
  propuesta: ["esperando", "esperando aprobación"],
  aprobada: ["cola", "en la cola"],
  "en-curso": ["pendiente", "corriendo"],
  probada: ["adoptado", "probada"],
  descartada: ["descartado", "descartada"],
};
const GRUPOS = [
  ["base", "Base"],
  ["fase 0", "Mirar los datos"],
  ["alta", "Prioridad alta"],
  ["media", "Prioridad media"],
  ["baja", "Prioridad baja"],
];

let D = null;

async function cargar() {
  try {
    const r = await fetch("data/proyecto.json", { cache: "no-store" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    D = await r.json();
  } catch (e) {
    $("#vista").innerHTML = `<p class="intro">No pude leer <code>data/proyecto.json</code> (${esc(e.message)}). Si abriste el archivo desde el disco, servilo con <code>python -m http.server</code> desde <code>docs/</code>.</p>`;
    return;
  }
  grafico();
  resumen();
  $("#pie").innerHTML = `Actualizado ${esc(D.generado)}. Código, logs y resultados en <a href="https://github.com/${esc(D.repo)}">github.com/${esc(D.repo)}</a>.`;
  const pendientes = D.ideas.filter((i) => i.estado === "propuesta").length;
  if (pendientes) $('.pestanas a[data-seccion="propuestas"]').insertAdjacentHTML("beforeend", `<span class="cuenta">${pendientes}</span>`);
  window.addEventListener("hashchange", ruta);
  ruta();
}

// ---------------------------------------------------------------- gráfico: un punto por experimento

function grafico() {
  const el = $("#grafico");
  const ex = D.experimentos.filter((e) => e.total != null);
  const W = Math.max(Math.min(el.clientWidth || 960, 960), 480), H = Math.round(W * 0.28) + 40, L = 62, R = 18, T = 18, B = 34;
  if (!ex.length) {
    el.innerHTML = `<svg viewBox="0 0 ${W} 90" role="img" aria-label="Todavía no hay experimentos"><text class="vacio" x="0" y="50">El primer asiento es la base (B0), medida en 2019: está corriendo.</text></svg>`;
    return;
  }
  const ys = ex.map((e) => e.total / 1e6);
  let y0 = Math.min(...ys), y1 = Math.max(...ys);
  const pad = Math.max((y1 - y0) * 0.18, 8);
  y0 -= pad; y1 += pad;
  const n = ex.length;
  const X = (i) => L + (n === 1 ? (W - L - R) / 2 : (i * (W - L - R)) / (n - 1));
  const Y = (v) => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);
  const paso = niceStep((y1 - y0) / 4);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Ganancia en test de cada experimento y del modelo vigente">`;
  for (let v = Math.ceil(y0 / paso) * paso; v <= y1; v += paso) {
    s += `<line class="eje" x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${L - 10}" y="${Y(v) + 4}" text-anchor="end">${nf0.format(Math.round(v))} M</text>`;
  }
  const base = ex.find((e) => e.estado === "base");
  if (base) s += `<line class="base-linea" x1="${L}" x2="${W - R}" y1="${Y(base.total / 1e6)}" y2="${Y(base.total / 1e6)}"/>`;
  let mejor = null, pts = [];
  ex.forEach((e, i) => {
    if (e.estado === "base" || e.estado === "confirmado") mejor = e.total / 1e6;
    if (mejor != null) pts.push([X(i), Y(mejor)]);
  });
  if (pts.length) {
    let d = `M${pts[0][0]},${pts[0][1]}`;
    for (let i = 1; i < pts.length; i++) d += ` H${pts[i][0]} V${pts[i][1]}`;
    if (n === 1) d += ` H${X(0) + 1}`;
    s += `<path class="escalon" d="${d}"/>`;
  }
  const cada = Math.max(1, Math.ceil(n / 12));
  ex.forEach((e, i) => {
    const [cl] = ESTADO_EXP[e.estado] || ["descartado"];
    s += `<circle class="punto ${cl}" data-id="${esc(e.id)}" cx="${X(i)}" cy="${Y(e.total / 1e6)}" r="${cl === "adoptado" ? 6 : 5}" tabindex="0" role="link" aria-label="${esc(e.id)}: ${esc(e.descripcion)}, ${M(e.total)}"/>`;
    if (i % cada === 0 || i === n - 1) s += `<text x="${X(i)}" y="${H - 10}" text-anchor="middle">${esc(e.id)}</text>`;
  });
  if (D.vigente) {
    const i = ex.findIndex((e) => e.id === D.vigente);
    if (i >= 0) s += `<text class="rotulo" x="${Math.min(X(i) + 10, W - R - 4)}" y="${Y(ex[i].total / 1e6) - 12}" text-anchor="${X(i) > W - 160 ? "end" : "start"}">vigente ${esc(ex[i].id)}</text>`;
  }
  el.innerHTML = s + "</svg>";
  const camino = $(".escalon", el);
  if (camino) camino.style.setProperty("--largo", Math.ceil(camino.getTotalLength()));
  const globo = $("#globo");
  el.querySelectorAll(".punto").forEach((c) => {
    const e = D.experimentos.find((x) => x.id === c.dataset.id);
    const mostrar = (ev) => {
      const r = c.getBoundingClientRect();
      globo.innerHTML = `<strong>${esc(e.id)}</strong> ${esc(e.descripcion)}<br>${M(e.total)} en los tres meses${e.comparacion ? `, ${signo(e.comparacion.delta_media * 3)} contra ${esc(e.comparacion.vigente)}` : ""}<br>${(ESTADO_EXP[e.estado] || ["", e.estado])[1]}`;
      globo.hidden = false;
      const x = Math.min(r.left + r.width / 2 + 12, innerWidth - globo.offsetWidth - 8);
      globo.style.left = `${x}px`;
      globo.style.top = `${r.top - globo.offsetHeight - 8 < 0 ? r.bottom + 8 : r.top - globo.offsetHeight - 8}px`;
    };
    c.addEventListener("mouseenter", mostrar);
    c.addEventListener("focus", mostrar);
    c.addEventListener("mouseleave", () => (globo.hidden = true));
    c.addEventListener("blur", () => (globo.hidden = true));
    const ir = () => (location.hash = `#/experimentos/${e.id}`);
    c.addEventListener("click", ir);
    c.addEventListener("keydown", (k) => (k.key === "Enter" || k.key === " ") && ir());
  });
}

function niceStep(x) {
  const p = Math.pow(10, Math.floor(Math.log10(x)));
  return [1, 2, 2.5, 5, 10].map((k) => k * p).find((k) => k >= x) || 10 * p;
}

function resumen() {
  const ex = D.experimentos.filter((e) => e.total != null);
  const vig = D.experimentos.find((e) => e.id === D.vigente);
  const base = D.experimentos.find((e) => e.estado === "base");
  const cola = D.ideas.filter((i) => i.estado === "aprobada").length;
  let t = "";
  if (vig) {
    t += `El modelo vigente es <a href="#/experimentos/${esc(vig.id)}">${esc(vig.id)}</a>, ${esc(vig.descripcion)}: sus listas habrían ganado <strong>${M(vig.total)}</strong> en ${D.escenario.meses_test.map((m) => nombreMes(m).toLowerCase()).join(", ").replace(/, ([^,]*)$/, " y $1")} de 2019`;
    if (base && base.id !== vig.id) t += `, ${signo(vig.total - base.total)} contra la base`;
    t += ". ";
  }
  t += `${ex.length} ${ex.length === 1 ? "experimento corrido" : "experimentos corridos"} y ${cola} ${cola === 1 ? "idea" : "ideas"} en la cola.`;
  $("#resumen").innerHTML = t;
}

// ---------------------------------------------------------------- rutas

function ruta() {
  const [, seccion = "experimentos", id = ""] = (location.hash || "#/experimentos").split("/");
  document.querySelectorAll(".pestanas a").forEach((a) => {
    const activa = a.dataset.seccion === (seccion === "proponer" ? "propuestas" : seccion);
    a.toggleAttribute("aria-current", activa);
    if (activa) a.setAttribute("aria-current", "page");
  });
  const vistas = { experimentos: vExperimentos, ideas: vIdeas, propuestas: vPropuestas, proponer: vPropuestas, aula: vAula, metodo: vMetodo, papers: vPapers };
  (vistas[seccion] || vExperimentos)(decodeURIComponent(id), seccion);
}

// ---------------------------------------------------------------- experimentos

function vExperimentos(abierto) {
  const ex = [...D.experimentos].reverse();
  const meses = D.escenario.meses_test.map(String);
  let h = `<h2>Experimentos</h2><p class="intro">Un renglón por corrida, el último arriba. Las columnas de los meses (de 2019) son la ganancia de la lista que habría entregado ese modelo; tocá un renglón para ver qué se cambió, cómo se midió, el código y los comentarios.</p>`;
  if (!ex.length) {
    vista(h + `<p class="intro">Todavía no terminó ninguno. El primero es la base, el modelo de la primera competencia llevado a estos datos.</p>`);
    return;
  }
  h += `<table class="libro fija"><colgroup><col style="width:66px"><col>${meses.map(() => '<col style="width:112px">').join("")}<col style="width:104px"><col style="width:118px"><col style="width:118px"><col style="width:104px"></colgroup><thead><tr><th>Asiento</th><th>Qué cambia</th>${meses.map((m) => `<th class="num">${NOMBRE_MES[m] || m}</th>`).join("")}<th class="num">Total</th><th class="num">Contra el vigente</th><th>Estado</th><th></th></tr></thead><tbody>`;
  for (const e of ex) {
    const [cl, nombre] = ESTADO_EXP[e.estado] || ["descartado", e.estado];
    const idea = D.ideas.find((i) => i.id === e.idea || (e.idea.startsWith("#") && i.issue === e.idea.slice(1)));
    const delta = e.comparacion ? e.comparacion.delta_media * meses.length : null;
    h += `<tr class="fila${abierto === e.id ? " abierta" : ""}" data-id="${esc(e.id)}" tabindex="0" aria-expanded="${abierto === e.id}">
      <td class="id">${esc(e.id)}</td>
      <td class="que">${esc(e.descripcion)}<small>${idea ? `${esc(idea.id)}, ${esc(idea.titulo)}` : esc(e.idea || "")} (${esc(e.fecha)})</small></td>
      ${meses.map((m) => `<td class="num" data-rotulo="${NOMBRE_MES[m]}">${e.meses[m] ? M(e.meses[m].lista) : "–"}</td>`).join("")}
      <td class="num" data-rotulo="Total"><strong>${M(e.total)}</strong></td>
      <td class="num ${claseSigno(delta)}" data-rotulo="Contra el vigente">${delta == null ? (e.estado === "base" ? '<span class="apagado">base</span>' : "–") : signo(delta)}</td>
      <td><span class="estado ${cl}">${nombre}</span></td>
      <td class="globo-com">${idea && idea.comentarios ? `${idea.comentarios} ${idea.comentarios === 1 ? "comentario" : "comentarios"}` : ""}</td>
    </tr>`;
    if (abierto === e.id) h += `<tr><td colspan="${meses.length + 6}" class="detalle-celda">${detalleExperimento(e, idea)}</td></tr>`;
  }
  vista(h + "</tbody></table>");
  document.querySelectorAll("tr.fila").forEach((tr) => {
    const ir = () => (location.hash = abierto === tr.dataset.id ? "#/experimentos" : `#/experimentos/${tr.dataset.id}`);
    tr.addEventListener("click", ir);
    tr.addEventListener("keydown", (k) => (k.key === "Enter" || k.key === " ") && (k.preventDefault(), ir()));
  });
  if (abierto) {
    const fila = $(`tr.fila[data-id="${CSS.escape(abierto)}"]`);
    if (fila) fila.scrollIntoView({ block: "start", behavior: "smooth" });
    const idea = D.ideas.find((i) => i.id === (D.experimentos.find((x) => x.id === abierto) || {}).idea);
    if (idea && idea.issue) comentarios(idea.issue, $("#comentarios"));
  }
}

function tablaMeses(meses, titulo) {
  const ms = Object.keys(meses);
  if (!ms.length) return "";
  const tot = ms.reduce((a, m) => a + meses[m].lista, 0);
  return `<table class="meses"><caption class="oculto">${esc(titulo)}</caption><thead><tr><th>${esc(titulo)}</th><th>Lista</th><th>Envíos</th><th>Aciertos</th><th>Meseta</th><th>AUC</th></tr></thead><tbody>
    ${ms.map((m) => `<tr><td>${NOMBRE_MES[m] || m}</td><td>${M(meses[m].lista)}</td><td>${N(meses[m].envios)}</td><td>${meses[m].aciertos == null ? "–" : `${N(meses[m].aciertos)} de ${N(meses[m].baja2)}`}</td><td>${M(meses[m].meseta)}</td><td>${A(meses[m].auc)}</td></tr>`).join("")}
    <tr class="total"><td>Total</td><td>${M(tot)}</td><td></td><td></td><td></td><td></td></tr></tbody></table>`;
}

function detalleExperimento(e, idea) {
  const meses = e.meses || {};
  const m0 = meses[Object.keys(meses)[0]] || {};
  let izq = `<dl>
    <dt>Qué se cambió</dt><dd>${esc(e.descripcion)}</dd>
    ${e.hipotesis ? `<dt>Por qué debería mejorar</dt><dd>${esc(e.hipotesis)}</dd>` : ""}
    ${idea ? `<dt>Idea</dt><dd><a href="#/ideas/${esc(idea.id)}">${esc(idea.id)}, ${esc(idea.titulo)}</a>. ${esc(idea.detalle)}</dd>` : ""}
    ${idea && idea.fuente ? `<dt>De dónde salió</dt><dd>${idea.fuente_url ? `<a href="${esc(idea.fuente_url)}">${esc(idea.fuente)}</a>` : esc(idea.fuente)}</dd>` : ""}
    <dt>Cómo se midió</dt><dd>${idea && idea.como_medir ? esc(idea.como_medir) + " " : ""}Cada mes entrenado con lo que se sabía entonces (${m0.train_meses ? `para julio, ${esc(m0.train_meses.join(", "))}` : "ventana de 6 meses"}), ${N(e.semillas ? e.semillas.length : null)} semillas, lista armada con el corte del modelo y medida con la clase verdadera.</dd>
    <dt>Contra el vigente</dt><dd>${e.comparacion ? `${signo(e.comparacion.delta_media * 3)} en los tres meses (gana en ${e.comparacion.meses_ganados} de 3; por semilla ${signo(e.comparacion.pareado_media * 3)} ± ${M(e.comparacion.pareado_se * 3)})` : e.estado === "base" ? "Es la base: el primer vigente." : "–"}</dd>
    ${e.confirmacion ? `<dt>Confirmación</dt><dd>${esc(e.confirmacion.decision)} con ${D.escenario.semillas_confirmacion} semillas nuevas: ${M(e.confirmacion.ensamble_a * 3)} contra ${M(e.confirmacion.ensamble_b * 3)}.</dd>` : ""}
    <dt>Tiempo</dt><dd>${e.segundos ? `${Math.round(e.segundos / 60)} minutos` : "–"}</dd>
  </dl>`;
  if (e.error) izq += `<pre class="error">${esc(e.error)}</pre>`;
  let der = tablaMeses(meses, "Data original");
  if (e.modificado) der += `<div style="height:18px"></div>` + tablaMeses(e.modificado.meses, "Data modificada");
  if (e.produccion) {
    der += `<h4 style="margin:22px 0 6px;font-weight:500">Qué habría pasado en 2020 (no decide nada)</h4><p class="nota" style="margin-top:0">El modelo congelado, reentrenado mes a mes con su ventana, sin cambios.</p>` + tablaMeses(e.produccion.original, "Producción 2020");
    if (Object.keys(e.produccion.modificado || {}).length) der += `<div style="height:14px"></div>` + tablaMeses(e.produccion.modificado, "Aula con data modificada");
  }
  der += `<p class="nota">Lista: lo que habría ganado la lista entregada. Meseta: lo mejor que se podía con ese orden si el corte fuera perfecto. Aciertos: BAJA+2 dentro de la lista, sobre el total del mes.</p>`;
  der += `<div class="enlaces"><a href="${esc(e.archivos.codigo)}">Código (train.py)</a>${e.archivos.diff ? `<a href="${esc(e.archivos.diff)}">Cambios contra el vigente</a>` : ""}<a href="${esc(e.archivos.log)}">Log de la corrida</a></div>`;
  const com = idea && idea.issue ? `<div class="comentarios"><h4>Comentarios</h4><div id="comentarios"><p class="nota">Cargando…</p></div>
    <p class="enlaces"><a href="https://github.com/${esc(D.repo)}/issues/${esc(idea.issue)}#new_comment_field">Comentar en GitHub</a></p></div>` : "";
  return `<div class="detalle"><div>${izq}</div><div>${der}</div>${com}</div>`;
}

// ---------------------------------------------------------------- ideas

function vIdeas(abierta) {
  let h = `<h2>Ideas</h2><p class="intro">Lo que está en la cola y lo que ya se probó, con la fuente de cada idea y cómo se mide. Las del plan salen de los papers y de lo que aprendimos en la simulación del aula; las de las compañeras entran cuando Emmanuel las aprueba.</p>`;
  const lista = D.ideas.filter((i) => i.estado !== "propuesta");
  for (const [clave, titulo] of GRUPOS) {
    const grupo = lista.filter((i) => i.prioridad === clave);
    if (!grupo.length) continue;
    h += `<section class="grupo-ideas"><h3>${titulo}</h3>${tablaIdeas(grupo, abierta)}</section>`;
  }
  const otras = lista.filter((i) => !GRUPOS.some(([c]) => c === i.prioridad));
  if (otras.length) h += `<section class="grupo-ideas"><h3>Otras</h3>${tablaIdeas(otras, abierta)}</section>`;
  vista(h);
  enganchar("ideas", abierta);
}

function tablaIdeas(ideas, abierta) {
  let h = `<table class="libro fija"><colgroup><col style="width:64px"><col><col style="width:27%"><col style="width:150px"><col style="width:112px"></colgroup><thead><tr><th>Idea</th><th>Qué y por qué</th><th>Fuente</th><th>Estado</th><th></th></tr></thead><tbody>`;
  for (const i of ideas) {
    const [cl, nombre] = ESTADO_IDEA[i.estado] || ["cola", i.estado];
    h += `<tr class="fila${abierta === i.id ? " abierta" : ""}" data-id="${esc(i.id)}" tabindex="0" aria-expanded="${abierta === i.id}">
      <td class="id">${esc(i.id)}</td>
      <td class="que">${esc(i.titulo)}<small>${esc(recortar(i.detalle, 150))}</small></td>
      <td data-rotulo="Fuente">${i.fuente_url ? `<a href="${esc(i.fuente_url)}">${esc(recortar(i.fuente, 60))}</a>` : esc(recortar(i.fuente, 60))}</td>
      <td><span class="estado ${cl}">${nombre}</span>${i.experimentos.length ? `<br><small class="apagado">${i.experimentos.map((x) => `<a href="#/experimentos/${esc(x)}">${esc(x)}</a>`).join(", ")}</small>` : ""}</td>
      <td class="globo-com">${i.comentarios ? `${i.comentarios} ${i.comentarios === 1 ? "comentario" : "comentarios"}` : ""}</td></tr>`;
    if (abierta === i.id) h += `<tr><td colspan="5" class="detalle-celda">${detalleIdea(i)}</td></tr>`;
  }
  return h + "</tbody></table>";
}

function detalleIdea(i) {
  const exps = i.experimentos.map((x) => D.experimentos.find((e) => e.id === x)).filter(Boolean);
  const izq = `<dl>
    <dt>Qué probar y por qué</dt><dd>${esc(i.detalle)}</dd>
    <dt>De dónde salió</dt><dd>${i.fuente_url ? `<a href="${esc(i.fuente_url)}">${esc(i.fuente)}</a>` : esc(i.fuente || "–")}</dd>
    <dt>Cómo se mide</dt><dd>${esc(i.como_medir || "Ganancia de las tres listas de test contra el vigente.")}</dd>
    <dt>Quién la propuso</dt><dd>${i.origen === "plan" ? "El plan inicial" : esc(i.origen)}</dd>
  </dl>`;
  const der = exps.length
    ? `<table class="meses"><thead><tr><th>Experimento</th><th>Total</th><th>Contra el vigente</th><th>Estado</th></tr></thead><tbody>${exps.map((e) => `<tr><td><a href="#/experimentos/${esc(e.id)}">${esc(e.id)}</a> ${esc(e.descripcion)}</td><td>${M(e.total)}</td><td class="${claseSigno(e.comparacion && e.comparacion.delta_media)}">${e.comparacion ? signo(e.comparacion.delta_media * 3) : "–"}</td><td>${(ESTADO_EXP[e.estado] || ["", e.estado])[1]}</td></tr>`).join("")}</tbody></table>`
    : `<p class="nota">Todavía no se corrió.</p>`;
  const com = i.issue ? `<div class="comentarios"><h4>Comentarios</h4><div id="comentarios"><p class="nota">Cargando…</p></div>
    <p class="enlaces"><a href="https://github.com/${esc(D.repo)}/issues/${esc(i.issue)}#new_comment_field">Comentar en GitHub</a></p></div>` : "";
  return `<div class="detalle"><div>${izq}</div><div>${der}</div>${com}</div>`;
}

function enganchar(seccion, abierta) {
  document.querySelectorAll("tr.fila").forEach((tr) => {
    const ir = () => (location.hash = abierta === tr.dataset.id ? `#/${seccion}` : `#/${seccion}/${tr.dataset.id}`);
    tr.addEventListener("click", (ev) => ev.target.closest("a") || ir());
    tr.addEventListener("keydown", (k) => (k.key === "Enter" || k.key === " ") && (k.preventDefault(), ir()));
  });
  if (abierta) {
    const fila = $(`tr.fila[data-id="${CSS.escape(abierta)}"]`);
    if (fila) fila.scrollIntoView({ block: "start", behavior: "smooth" });
    const i = D.ideas.find((x) => x.id === abierta);
    if (i && i.issue) comentarios(i.issue, $("#comentarios"));
  }
}

// ---------------------------------------------------------------- propuestas

function vPropuestas(abierta, seccion) {
  const pend = D.ideas.filter((i) => i.estado === "propuesta");
  let h = `<h2>Propuestas</h2><p class="intro">Cualquiera del grupo puede sumar un experimento: algo de un paper, de una clase o una idea propia. Así sigue:</p>
    <ol class="pasos"><li>Completás el formulario de abajo; se abre GitHub con la propuesta ya escrita y la publicás (hace falta una cuenta de GitHub).</li>
    <li>Emmanuel la revisa. Si la aprueba, entra a la cola de ideas.</li>
    <li>Se corre con el mismo protocolo que el resto y el resultado aparece en el libro y como comentario en la propuesta.</li></ol>`;
  h += `<h3>Esperando aprobación</h3>`;
  h += pend.length ? tablaIdeas(pend, abierta) : `<p class="intro" id="sin-pendientes">No hay propuestas esperando.</p>`;
  h += `<div id="en-vivo"></div>`;
  h += `<h3 id="proponer">Proponer un experimento</h3>
  <form class="propuesta-form" id="form-propuesta">
    <label>Título<input name="titulo" required maxlength="120" placeholder="Por ejemplo: sacar del entrenamiento el mes de junio"></label>
    <label>Qué probar <small>Qué cambiar en el modelo o en el proceso.</small><textarea name="que" required></textarea></label>
    <label>Por qué debería mejorar <small>Qué esperás ver y en qué meses.</small><textarea name="porque"></textarea></label>
    <label>Fuente <small>El link al paper, la clase de donde salió, o "idea propia" y por qué se te ocurrió.</small><input name="fuente" placeholder="https://arxiv.org/abs/..."></label>
    <label>Cómo medirlo <small>Si querés mirar algo además de la ganancia de las tres listas de test (201908 a 201910).</small><textarea name="medir"></textarea></label>
    <label>Prioridad<select name="prioridad"><option>alta</option><option selected>media</option><option>baja</option></select></label>
    <div class="acciones"><button class="boton" type="submit">Abrir la propuesta en GitHub</button><span class="nota">Se abre en otra pestaña; nada se publica hasta que la confirmes ahí.</span></div>
  </form>`;
  vista(h);
  enganchar("propuestas", abierta);
  $("#form-propuesta").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const f = new FormData(ev.target);
    const q = new URLSearchParams({ template: "propuesta.yml", title: `[propuesta] ${f.get("titulo")}`, que: f.get("que"), porque: f.get("porque"), fuente: f.get("fuente"), medir: f.get("medir"), prioridad: f.get("prioridad") });
    window.open(`https://github.com/${D.repo}/issues/new?${q}`, "_blank", "noopener");
  });
  if (seccion === "proponer") $("#proponer").scrollIntoView({ block: "start" });
  propuestasEnVivo(pend);
}

async function propuestasEnVivo(conocidas) {
  try {
    const r = await fetch(`https://api.github.com/repos/${D.repo}/issues?labels=propuesta&state=open&per_page=50`);
    if (!r.ok) return;
    const nuevas = (await r.json()).filter((it) => !it.pull_request && !(it.labels || []).some((l) => l.name === "aprobada") && !conocidas.some((c) => c.issue === String(it.number)));
    if (!nuevas.length) return;
    const sin = $("#sin-pendientes");
    if (sin) sin.remove();
    $("#en-vivo").innerHTML = `<table class="libro"><tbody>${nuevas.map((it) => `<tr><td class="id">#${it.number}</td><td class="que"><a href="${esc(it.html_url)}">${esc(it.title.replace(/^\[propuesta\]\s*/i, ""))}</a><small>de ${esc(it.user.login)}, recién llegada: todavía no está en el libro</small></td><td><span class="estado esperando">esperando aprobación</span></td></tr>`).join("")}</tbody></table>`;
  } catch (_) { /* sin conexión con GitHub: se muestra lo que trae el libro */ }
}

// ---------------------------------------------------------------- comentarios en vivo desde GitHub

async function comentarios(issue, el) {
  if (!el) return;
  try {
    const r = await fetch(`https://api.github.com/repos/${D.repo}/issues/${issue}/comments?per_page=100`);
    if (r.status === 403) { el.innerHTML = `<p class="nota">GitHub limita las consultas sin cuenta; los comentarios se ven en el issue.</p>`; return; }
    if (!r.ok) throw new Error(r.status);
    const cs = await r.json();
    el.innerHTML = cs.length
      ? cs.map((c) => `<div class="comentario"><div class="quien">${esc(c.user.login)}<span>${new Date(c.created_at).toLocaleDateString("es-AR", { day: "numeric", month: "long", hour: "2-digit", minute: "2-digit" })}</span></div><div class="texto">${md(c.body)}</div></div>`).join("")
      : `<p class="nota">Nadie comentó todavía.</p>`;
  } catch (_) {
    el.innerHTML = `<p class="nota">No pude traer los comentarios; están en el issue.</p>`;
  }
}

function md(t) {
  return esc(t || "")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(https?:\/\/[^\s<)]+)/g, '<a href="$1">$1</a>')
    .split(/\n{2,}/).map((p) => `<p>${p.replace(/\n/g, "<br>")}</p>`).join("");
}

// ---------------------------------------------------------------- método y papers

function vAula() {
  const Au = D.aula || {};
  const ex = (id) => D.experimentos.find((e) => e.id === id);
  const fila = (nombre, e) => {
    if (!e) return "";
    const o = e.meses || {}, m = (e.modificado || {}).meses || {};
    const ms = Object.keys(o);
    return `<tr><td class="que"><a href="#/experimentos/${esc(e.id)}">${esc(e.id)}</a> ${esc(nombre)}<small>${esc(e.descripcion)}</small></td>${ms.map((k) => `<td class="num">${M(o[k].lista)}<br><span class="apagado">${m[k] ? M(m[k].lista) : "–"}</span></td>`).join("")}<td class="num"><strong>${M(e.total)}</strong><br><span class="apagado">${e.modificado ? M(Object.values(m).reduce((t, x) => t + x.lista, 0)) : "–"}</span></td></tr>`;
  };
  const mod = ex(Au.modelo), alt = ex(Au.alternativa);
  const ref = D.experimentos.find((e) => e.modificado && e.descripcion.startsWith("el vigente"));
  const ms = mod ? Object.keys(mod.meses) : D.escenario.meses_test.map(String);
  vista(`<h2>Para el aula</h2><div class="texto-largo"><p>${esc(Au.texto || "")}</p>
    <p><strong>Por qué ese umbral.</strong> En los meses normales de 2019 hay entre ${Au.saltos_2019 ? Au.saltos_2019.normal.join(" y ") : "–"} montos con salto contra el mes anterior; en el primer mes de la data modificada simulada, ${Au.saltos_2019 ? Au.saltos_2019.modificado_primer_mes : "–"}. La regla se activa sólo en el mes en que la escala cambia de golpe, que es donde la alternativa recupera plata.</p></div>
    <h3>Cómo aguanta cada uno la data modificada</h3><p class="intro">Arriba, la ganancia con la data original de 2019; abajo, en gris, con la data modificada simulada sobre esos mismos meses.</p>
    <table class="libro"><thead><tr><th>Modelo</th>${ms.map((k) => `<th class="num">${nombreMes(k)}</th>`).join("")}<th class="num">Total</th></tr></thead><tbody>
    ${fila("modelo vigente, medido con la data modificada", ref)}${fila("alternativa robusta", alt)}</tbody></table>
    <p class="nota">En el aula, la herramienta calcula las dos listas cada mes, entrega la que diga la regla y deja cambiarla a mano sin reentrenar. Nada de esto se eligió mirando 2020.</p>
    ${Au.preparacion ? `<h3>Qué pasó después (informe: no decide nada)</h3>
    <p class="intro">Con el modelo ya congelado, se lo reentrenó mes a mes en 2020 como en producción, y se ensayó el aula con cronómetro (unos ${String(Au.ensayo.minutos_por_mes).replace(".", ",")} minutos por mes). Estos números se miraron recién al final y no cambiaron nada del modelo.</p>
    <table class="libro"><thead><tr><th>Mes</th><th class="num">Envíos</th><th class="num">Aciertos</th><th class="num">BAJA+2</th><th class="num">Ganancia</th></tr></thead><tbody>
    ${Au.preparacion.map((f) => `<tr><td>${nombreMes(f[0])} ${f[0].slice(0, 4)}</td><td class="num">${N(f[1])}</td><td class="num">${N(f[2])}</td><td class="num">${N(f[3])}</td><td class="num">${nf1.format(f[4])} M</td></tr>`).join("")}
    <tr class="total"><td>Preparación</td><td></td><td></td><td></td><td class="num">${nf1.format(Au.preparacion.reduce((t, f) => t + f[4], 0))} M</td></tr></tbody></table>
    <h3>El ensayo del aula</h3><table class="libro"><thead><tr><th>Mes</th><th>Data original</th><th class="num">Ganancia</th><th>Data modificada</th><th class="num">Ganancia</th><th class="num">Siempre el modelo</th></tr></thead><tbody>
    ${Au.ensayo.original.map((o, i) => { const m = Au.ensayo.modificado[i]; return `<tr><td>${nombreMes(o[0])} ${o[0].slice(0, 4)}</td><td>${o[1]} <span class="apagado">(${o[2]} saltos)</span></td><td class="num">${nf1.format(o[3])} M</td><td>${m[1]} <span class="apagado">(${m[2]} saltos)</span></td><td class="num">${nf1.format(m[3])} M</td><td class="num apagado">${nf1.format(m[4])} M</td></tr>`; }).join("")}
    <tr class="total"><td>Total</td><td></td><td class="num">${nf1.format(Au.ensayo.original.reduce((t, f) => t + f[3], 0))} M</td><td></td><td class="num">${nf1.format(Au.ensayo.modificado.reduce((t, f) => t + f[3], 0))} M</td><td class="num">${nf1.format(Au.ensayo.modificado.reduce((t, f) => t + f[4], 0))} M</td></tr></tbody></table>` : ""}`);
}

function vMetodo() {
  const E = D.escenario;
  const mt = E.meses_test.map((m) => nombreMes(m).toLowerCase());
  vista(`<h2>Cómo medimos</h2><div class="texto-largo">
    <p><strong>La regla.</strong> El modelo se define sólo con datos de 2019. El presente es diciembre de 2019: se ven los datos hasta ese mes y la clase como se la sabía entonces (completa hasta octubre, noviembre sólo con las BAJA+1, diciembre sin clase). En 2020 el modelo sólo se reentrena todos los meses con los últimos ${E.ventana} meses, sin cambiarlo. En julio, agosto y septiembre de 2020, el aula, se implementan los cambios que hagan falta con la data que llega.</p>
    <p><strong>Train y test.</strong> El test son ${mt.join(", ").replace(/, ([^,]*)$/, " y $1")} de 2019, los tres últimos meses con clase completa en diciembre. Para cada uno, el modelo se entrena con lo que se sabía en ese momento, con la misma regla que en producción (clase completa hasta dos meses antes y, del mes anterior, sólo quién se dio de baja), arma su lista con su propia regla de corte y la lista se mide con la clase.</p>
    <p><strong>La ganancia.</strong> Cada BAJA+2 estimulado suma <span class="cifra">${N(E.ganancia)}</span> y cada estímulo cuesta <span class="cifra">${N(E.costo)}</span>. Lo que decide es la suma de las tres listas. También se informan la meseta de la curva (lo mejor que se podía con ese orden si el corte fuera perfecto) y la AUC.</p>
    <p><strong>Cuándo se adopta un cambio.</strong> Primero con ${E.semillas_screening} semillas: pasa a confirmación si la ganancia media por mes mejora más de ${M(E.umbral_screening)} y gana en al menos ${E.min_meses_ganados} de los 3 meses. Después con ${E.semillas_confirmacion} semillas nuevas: se adopta si la diferencia por semilla supera dos errores estándar. A igual ganancia gana el código más simple.</p>
    <p><strong>La data modificada.</strong> En el aula los meses llegan alterados: variables rotas, drifting, cosas con la data. Para prepararse sin mirar 2020, a los que se confirman se los corre también con esos mismos tres meses de 2019 alterados (montos inflados de golpe, variables en cero o vacías, columnas que faltan).</p>
    <p><strong>2020 no decide nada.</strong> Cuando el modelo quede congelado, se lo reentrena mes a mes en 2020 y se informa qué habría ganado. Ningún experimento se elige mirando ese número.</p>
    <p>Cada corrida deja su código exacto, el cambio contra el vigente y su log en el repositorio. Lo aprendido se anota en <a href="https://github.com/${esc(D.repo)}/blob/main/LECCIONES.md">LECCIONES.md</a>. Dos corridas viejas, medidas por error en 2020, quedaron en <a href="https://github.com/${esc(D.repo)}/tree/main/anulados">anulados/</a> y no cuentan.</p>
    ${D.lecciones.length ? `<h3>Lo aprendido hasta ahora</h3><ul>${D.lecciones.map((l) => `<li>${esc(l)}</li>`).join("")}</ul>` : ""}
  </div>`);
}

function vPapers() {
  let h = `<h2>Papers</h2><p class="intro">Lo que se leyó, si sirve para este problema y en qué idea se usa.</p><table class="libro"><thead><tr><th>Paper</th><th>Para qué</th><th>Veredicto</th><th>Ideas</th></tr></thead><tbody>`;
  for (const p of D.papers) {
    h += `<tr><td class="que"><a href="${esc(p.url)}">${esc(p.titulo)}</a><small>prioridad ${esc(p.prioridad)}</small></td><td>${esc(p.para)}</td><td><span class="estado ${p.veredicto.startsWith("no") ? "descartado" : "adoptado"}">${esc(p.veredicto)}</span></td><td>${p.ideas.map((i) => `<a href="#/ideas/${esc(i)}">${esc(i)}</a>`).join(", ") || "–"}</td></tr>`;
  }
  vista(h + "</tbody></table>");
}

// ---------------------------------------------------------------- utilidades

function recortar(t, n) {
  t = String(t || "");
  return t.length > n ? t.slice(0, n - 1).trimEnd() + "…" : t;
}

function vista(html) {
  const v = $("#vista");
  v.innerHTML = html;
}

cargar();
