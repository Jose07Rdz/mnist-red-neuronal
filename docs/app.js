import { cargarModelo, pasarCNN, pasarMLP } from "./red.js";

const datos = await fetch("modelos.json").then((r) => r.json());
const { media, desviacion } = datos.normalizacion;
const modelos = { mlp: cargarModelo(datos.mlp), cnn: cargarModelo(datos.cnn) };
let modeloActivo = "cnn";

const lienzo = document.getElementById("lienzo");
const ctx = lienzo.getContext("2d", { willReadFrequently: true });
const red = document.getElementById("red");
const ANCHO = 960, ALTO = 520;

// ---------- Dibujo ----------

function limpiarLienzo() {
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, lienzo.width, lienzo.height);
}

let dibujando = false, ultimo = null;

function punto(e) {
  const r = lienzo.getBoundingClientRect();
  return [((e.clientX - r.left) / r.width) * lienzo.width, ((e.clientY - r.top) / r.height) * lienzo.height];
}

function trazo(a, b) {
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 20;
  ctx.lineCap = ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(...a);
  ctx.lineTo(...b);
  ctx.stroke();
}

lienzo.addEventListener("pointerdown", (e) => {
  dibujando = true;
  lienzo.setPointerCapture(e.pointerId);
  ultimo = punto(e);
  trazo(ultimo, ultimo);
  pedirPrediccion();
});
lienzo.addEventListener("pointermove", (e) => {
  if (!dibujando) return;
  const p = punto(e);
  trazo(ultimo, p);
  ultimo = p;
  pedirPrediccion();
});
for (const ev of ["pointerup", "pointercancel"]) lienzo.addEventListener(ev, () => (dibujando = false));

document.getElementById("borrar").addEventListener("click", () => {
  limpiarLienzo();
  actualizar();
});

for (const b of document.querySelectorAll(".selector button")) {
  b.addEventListener("click", () => {
    modeloActivo = b.dataset.modelo;
    for (const o of document.querySelectorAll(".selector button")) o.setAttribute("aria-selected", o === b);
    actualizar();
  });
}

let pendiente = false;
function pedirPrediccion() {
  if (pendiente) return;
  pendiente = true;
  requestAnimationFrame(() => {
    pendiente = false;
    actualizar();
  });
}

// ---------- Preparar la imagen como en MNIST ----------
// MNIST: el dígito cabe en una caja de 20×20 y su centro de masa queda en el centro de 28×28.

function prepararImagen() {
  const { width: w, height: h } = lienzo;
  const px = ctx.getImageData(0, 0, w, h).data;
  const gris = new Float32Array(w * h);
  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const v = px[(y * w + x) * 4] / 255;
      gris[y * w + x] = v;
      if (v > 0.05) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  if (x1 < 0) return null;

  // Recortar y escalar a 20 px en el lado mayor, promediando el área de cada píxel.
  const bw = x1 - x0 + 1, bh = y1 - y0 + 1;
  const escala = 20 / Math.max(bw, bh);
  const tw = bw * escala, th = bh * escala;
  const ox = (28 - tw) / 2, oy = (28 - th) / 2;
  const img = new Float32Array(784);
  const M = 6; // muestras por eje dentro de cada píxel
  for (let y = 0; y < 28; y++)
    for (let x = 0; x < 28; x++) {
      let s = 0;
      for (let sy = 0; sy < M; sy++)
        for (let sx = 0; sx < M; sx++) {
          const fx = (x + (sx + 0.5) / M - ox) / escala + x0;
          const fy = (y + (sy + 0.5) / M - oy) / escala + y0;
          if (fx >= x0 && fx <= x1 + 1 && fy >= y0 && fy <= y1 + 1)
            s += gris[Math.min(h - 1, Math.floor(fy)) * w + Math.min(w - 1, Math.floor(fx))];
        }
      img[y * 28 + x] = s / (M * M);
    }

  // Mover el centro de masa al centro.
  let masa = 0, cx = 0, cy = 0;
  for (let i = 0; i < 784; i++) {
    masa += img[i];
    cx += (i % 28) * img[i];
    cy += Math.floor(i / 28) * img[i];
  }
  const dx = Math.round(13.5 - cx / masa), dy = Math.round(13.5 - cy / masa);
  const centrada = new Float32Array(784);
  for (let y = 0; y < 28; y++)
    for (let x = 0; x < 28; x++) {
      const xs = x - dx, ys = y - dy;
      if (xs >= 0 && xs < 28 && ys >= 0 && ys < 28) centrada[y * 28 + x] = img[ys * 28 + xs];
    }
  return centrada;
}

// ---------- Predicción ----------

const barras = document.getElementById("barras");
for (let d = 0; d < 10; d++)
  barras.insertAdjacentHTML("beforeend",
    `<li><span class="num">${d}</span><span class="pista"><span class="relleno"></span></span><span class="pct">0%</span></li>`);

function actualizar() {
  const modelo = modelos[modeloActivo];
  document.getElementById("precision").textContent =
    `${modeloActivo === "cnn" ? "Red convolucional" : "Red densa"}: ${(modelo.precision * 100).toFixed(1)} % de aciertos en las 10,000 imágenes de prueba de MNIST`;

  const imagen = prepararImagen();
  const salida = imagen
    ? (modeloActivo === "cnn" ? pasarCNN : pasarMLP)(modelo, imagen.map((v) => (v - media) / desviacion))
    : null;

  const filas = barras.children;
  const ganador = salida ? salida.probabilidades.indexOf(Math.max(...salida.probabilidades)) : -1;
  for (let d = 0; d < 10; d++) {
    const p = salida ? salida.probabilidades[d] : 0;
    filas[d].querySelector(".relleno").style.width = `${p * 100}%`;
    filas[d].querySelector(".pct").textContent = `${(p * 100).toFixed(p > 0.995 || p < 0.005 ? 0 : 1)}%`;
    filas[d].classList.toggle("gana", d === ganador);
  }
  document.getElementById("digito").textContent = salida ? ganador : "?";
  document.getElementById("seguridad").textContent = salida
    ? `Seguridad: ${(salida.probabilidades[ganador] * 100).toFixed(1)} %`
    : "Esperando tu dibujo";

  dibujarRed(imagen, salida);
}

// ---------- Diagrama de la red ----------

function color(nombre) {
  return getComputedStyle(document.documentElement).getPropertyValue(nombre).trim();
}

function rgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function mezcla(a, b, t) {
  t = Math.max(0, Math.min(1, t));
  return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(",")})`;
}

function maximo(arr) {
  let m = 0;
  for (const v of arr) if (v > m) m = v;
  return m || 1;
}

function columnaNeuronas(n, x, radioMax) {
  const espacio = Math.min((ALTO - 90) / n, radioMax * 2.6);
  const y0 = 60 + (ALTO - 90 - espacio * (n - 1)) / 2;
  const radio = Math.min(radioMax, espacio * 0.42);
  return Array.from({ length: n }, (_, i) => ({ x, y: y0 + i * espacio, r: radio }));
}

function dibujarMapa(g, valores, lado, x, y, tam, max, c0, c1) {
  const p = tam / lado;
  for (let i = 0; i < lado * lado; i++) {
    g.fillStyle = mezcla(c0, c1, valores[i] / max);
    g.fillRect(x + (i % lado) * p, y + Math.floor(i / lado) * p, Math.ceil(p), Math.ceil(p));
  }
}

function dibujarRed(imagen, salida) {
  const dpr = window.devicePixelRatio || 1;
  red.width = ANCHO * dpr;
  red.height = ALTO * dpr;
  const g = red.getContext("2d");
  g.scale(dpr, dpr);

  const cTexto = color("--texto-suave"), cFuerte = color("--texto");
  const c0 = rgb(color("--neurona")), c1 = rgb(color("--acento")), cLinea = color("--linea");
  const negro = [0, 0, 0], blanco = [255, 255, 255];
  g.font = "500 13px Inter, system-ui, sans-serif";
  g.textAlign = "center";
  g.textBaseline = "middle";

  const etiqueta = (texto, x) => {
    g.fillStyle = cTexto;
    g.fillText(texto, x, 22);
  };
  const lineas = (desde, hacia, intensidad) => {
    g.strokeStyle = cLinea;
    g.lineWidth = 1;
    for (const [i, j, t] of intensidad) {
      if (t < 0.12) continue;
      g.globalAlpha = Math.min(0.9, t * 0.8);
      g.beginPath();
      g.moveTo(desde[i].x, desde[i].y);
      g.lineTo(hacia[j].x, hacia[j].y);
      g.stroke();
    }
    g.globalAlpha = 1;
  };
  // Conexiones densas: se dibuja la contribución positiva peso × activación de cada neurona de origen.
  const contribuciones = (capa, act, nOrigen, nDestino) => {
    const res = [];
    let max = 0;
    for (let j = 0; j < nDestino; j++)
      for (let i = 0; i < nOrigen; i++) {
        const c = capa.pesos[j * nOrigen + i] * act[i];
        if (c > 0) {
          res.push([i, j, c]);
          if (c > max) max = c;
        }
      }
    return res.map(([i, j, c]) => [i, j, c / (max || 1)]);
  };
  const neuronas = (pos, act) => {
    const max = act ? maximo(act) : 1;
    pos.forEach((p, i) => {
      g.fillStyle = mezcla(c0, c1, act ? act[i] / max : 0);
      g.beginPath();
      g.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      g.fill();
    });
  };

  // Entrada 28×28
  const tamEntrada = 140, xEntrada = 30, yEntrada = (ALTO - tamEntrada) / 2 + 15;
  etiqueta("Entrada 28×28", xEntrada + tamEntrada / 2);
  dibujarMapa(g, imagen || new Float32Array(784), 28, xEntrada, yEntrada, tamEntrada, 1, negro, blanco);
  const bordeEntrada = Array.from({ length: 8 }, (_, i) => ({ x: xEntrada + tamEntrada, y: yEntrada + (i + 0.5) * tamEntrada / 8 }));

  const modelo = modelos[modeloActivo];
  const xSalida = 860;
  const salidaPos = columnaNeuronas(10, xSalida, 17);

  if (modeloActivo === "mlp") {
    const [, c2L, c3L] = modelo.capas;
    const p1 = columnaNeuronas(64, 380, 6), p2 = columnaNeuronas(32, 620, 8);
    const [h1, h2] = salida ? salida.capas : [null, null];
    etiqueta("64 neuronas", 380);
    etiqueta("32 neuronas", 620);
    if (salida) {
      const max1 = maximo(h1);
      lineas(bordeEntrada, p1, p1.flatMap((_, j) => bordeEntrada.map((_, i) => [i, j, (h1[j] / max1) * 0.5])));
      lineas(p1, p2, contribuciones(c2L, h1, 64, 32));
    }
    neuronas(p1, h1);
    if (salida) lineas(p2, salidaPos, contribuciones(c3L, h2, 32, 10));
    neuronas(p2, h2);
    explicar("La red densa aplana la imagen en 784 números. Cada una de las 64 neuronas de la primera capa mira todos los píxeles a la vez; las líneas muestran qué neuronas empujan más hacia la siguiente capa.");
  } else {
    const [c1L, c2L, , c4L] = modelo.capas;
    const n1 = c1L.forma[0], n2 = c2L.forma[0];
    const [m1, m2] = salida ? salida.mapas : [new Float32Array(n1 * 196), new Float32Array(n2 * 49)];
    const max1 = maximo(m1), max2 = maximo(m2);

    // 8 mapas de 14×14
    const tam1 = 46, sep1 = (ALTO - 90) / n1, x1 = 250;
    etiqueta(`${n1} filtros · 14×14`, x1 + tam1 / 2);
    const pos1 = Array.from({ length: n1 }, (_, i) => ({ x: x1, y: 60 + i * sep1 + (sep1 - tam1) / 2 }));
    // 16 mapas de 7×7 en dos columnas
    const tam2 = 30, sep2 = (ALTO - 90) / (n2 / 2), x2 = 400;
    etiqueta(`${n2} filtros · 7×7`, x2 + tam2 + 4);
    const pos2 = Array.from({ length: n2 }, (_, i) => ({ x: x2 + (i % 2) * (tam2 + 8), y: 60 + Math.floor(i / 2) * sep2 + (sep2 - tam2) / 2 }));

    const promedio = (m, i, n) => m.subarray(i * n, (i + 1) * n).reduce((a, b) => a + b, 0) / n;
    const act1 = pos1.map((_, i) => promedio(m1, i, 196)), act2 = pos2.map((_, i) => promedio(m2, i, 49));
    const ma1 = maximo(act1), ma2 = maximo(act2);
    const centro = (p, t) => ({ x: p.x + t / 2, y: p.y + t / 2 });
    const izq1 = pos1.map((p) => ({ x: p.x, y: p.y + tam1 / 2 })), der1 = pos1.map((p) => ({ x: p.x + tam1, y: p.y + tam1 / 2 }));
    const izq2 = pos2.map((p) => ({ x: p.x, y: p.y + tam2 / 2 }));
    if (salida) {
      lineas(bordeEntrada, izq1, izq1.flatMap((_, j) => bordeEntrada.map((_, i) => [i, j, (act1[j] / ma1) * 0.3])));
      lineas(der1, izq2, izq2.flatMap((_, j) => der1.map((_, i) => [i, j, (act1[i] / ma1) * (act2[j] / ma2) * 0.5])));
    }
    pos1.forEach((p, i) => dibujarMapa(g, m1.subarray(i * 196, (i + 1) * 196), 14, p.x, p.y, tam1, max1, c0, c1));
    pos2.forEach((p, i) => dibujarMapa(g, m2.subarray(i * 49, (i + 1) * 49), 7, p.x, p.y, tam2, max2, c0, c1));

    const pH = columnaNeuronas(64, 640, 6);
    const h = salida ? salida.capas[0] : null;
    etiqueta("64 neuronas", 640);
    if (salida) {
      const der2 = pos2.map((p) => centro(p, tam2)).map((p) => ({ x: p.x + tam2 / 2, y: p.y }));
      const maxH = maximo(h);
      lineas(der2, pH, pH.flatMap((_, j) => der2.map((_, i) => [i, j, (h[j] / maxH) * (act2[i] / ma2) * 0.3])));
      lineas(pH, salidaPos, contribuciones(c4L, h, 64, 10));
    }
    neuronas(pH, h);
    explicar("La red convolucional pasa filtros de 3×3 sobre la imagen: cada cuadro es lo que detectó un filtro (bordes, curvas, trazos en cierta dirección). La segunda capa combina esos patrones en formas más grandes antes de llegar a las neuronas densas.");
  }

  // Salida
  etiqueta("Salida", xSalida);
  const ganador = salida ? salida.probabilidades.indexOf(Math.max(...salida.probabilidades)) : -1;
  salidaPos.forEach((p, d) => {
    const prob = salida ? salida.probabilidades[d] : 0;
    g.fillStyle = mezcla(c0, c1, prob);
    g.beginPath();
    g.arc(p.x, p.y, p.r, 0, Math.PI * 2);
    g.fill();
    if (d === ganador) {
      g.strokeStyle = cFuerte;
      g.lineWidth = 2;
      g.stroke();
    }
    g.fillStyle = prob > 0.5 ? "#fff" : cFuerte;
    g.font = "600 14px 'JetBrains Mono', ui-monospace, monospace";
    g.fillText(d, p.x, p.y + 1);
    g.fillStyle = cTexto;
    g.font = "500 12px Inter, system-ui, sans-serif";
    g.textAlign = "left";
    if (salida) g.fillText(`${Math.round(prob * 100)}%`, p.x + p.r + 8, p.y + 1);
    g.textAlign = "center";
  });
}

function explicar(texto) {
  document.getElementById("explicacion").textContent = texto;
}

limpiarLienzo();
actualizar();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", actualizar);
document.fonts?.ready.then(actualizar);
