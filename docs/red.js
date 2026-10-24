// Cálculo de las redes en el navegador: mismas operaciones que PyTorch, sin librerías.

function base64AFloat32(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

export function cargarModelo(json) {
  return {
    precision: json.precision,
    capas: json.capas.map((c) => ({
      tipo: c.tipo,
      forma: c.forma,
      pesos: base64AFloat32(c.pesos),
      sesgo: base64AFloat32(c.sesgo),
    })),
  };
}

export { base64AFloat32 };

// out[o] = sesgo[o] + Σ pesos[o, i] · x[i]
function lineal(x, capa) {
  const [nSal, nEnt] = capa.forma;
  const out = new Float32Array(nSal);
  for (let o = 0; o < nSal; o++) {
    let s = capa.sesgo[o];
    const base = o * nEnt;
    for (let i = 0; i < nEnt; i++) s += capa.pesos[base + i] * x[i];
    out[o] = s;
  }
  return out;
}

// Convolución 3×3 con padding 1. x tiene forma [canales, alto, ancho].
function convolucion(x, canales, alto, ancho, capa) {
  const [nSal, nEnt, k] = capa.forma;
  const out = new Float32Array(nSal * alto * ancho);
  const r = (k - 1) / 2;
  for (let o = 0; o < nSal; o++) {
    for (let y = 0; y < alto; y++) {
      for (let xx = 0; xx < ancho; xx++) {
        let s = capa.sesgo[o];
        for (let c = 0; c < nEnt; c++) {
          for (let dy = 0; dy < k; dy++) {
            const yy = y + dy - r;
            if (yy < 0 || yy >= alto) continue;
            for (let dx = 0; dx < k; dx++) {
              const xs = xx + dx - r;
              if (xs < 0 || xs >= ancho) continue;
              s += capa.pesos[((o * nEnt + c) * k + dy) * k + dx] * x[(c * alto + yy) * ancho + xs];
            }
          }
        }
        out[(o * alto + y) * ancho + xx] = s;
      }
    }
  }
  return out;
}

function relu(x) {
  return x.map((v) => (v > 0 ? v : 0));
}

function maxPool2(x, canales, alto, ancho) {
  const h = alto / 2, w = ancho / 2;
  const out = new Float32Array(canales * h * w);
  for (let c = 0; c < canales; c++)
    for (let y = 0; y < h; y++)
      for (let xx = 0; xx < w; xx++) {
        const i = (c * alto + 2 * y) * ancho + 2 * xx;
        out[(c * h + y) * w + xx] = Math.max(x[i], x[i + 1], x[i + ancho], x[i + ancho + 1]);
      }
  return out;
}

export function softmax(logits) {
  const m = Math.max(...logits);
  const e = logits.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

// Devuelve las activaciones de cada capa para poder dibujarlas.
export function pasarMLP(modelo, entrada) {
  const [c1, c2, c3] = modelo.capas;
  const h1 = relu(lineal(entrada, c1));
  const h2 = relu(lineal(h1, c2));
  const logits = lineal(h2, c3);
  return { capas: [h1, h2], logits, probabilidades: softmax(logits) };
}

export function pasarCNN(modelo, entrada) {
  const [c1, c2, c3, c4] = modelo.capas;
  const m1 = maxPool2(relu(convolucion(entrada, 1, 28, 28, c1)), c1.forma[0], 28, 28);
  const m2 = maxPool2(relu(convolucion(m1, c1.forma[0], 14, 14, c2)), c2.forma[0], 14, 14);
  const h = relu(lineal(m2, c3));
  const logits = lineal(h, c4);
  return { mapas: [m1, m2], capas: [h], logits, probabilidades: softmax(logits) };
}
