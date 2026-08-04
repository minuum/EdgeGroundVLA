import fs from 'fs';
import { JSDOM } from 'jsdom';

const dom = new JSDOM('<!doctype html><body><div id="c"></div></body>', { pretendToBeVisual: true });
global.window = dom.window;
global.document = dom.window.document;
Object.defineProperty(global, 'navigator', { value: dom.window.navigator, configurable: true });
for (const k of ['SVGElement','Element','HTMLElement','DOMParser','MutationObserver','CSSStyleSheet','CSSStyleDeclaration','Node','NodeList','getComputedStyle','requestAnimationFrame']) {
  if (dom.window[k]) global[k] = dom.window[k];
}
// jsdom은 getBBox/getComputedTextLength 미구현 → 텍스트 크기 stub
dom.window.SVGElement.prototype.getBBox = function () {
  const t = (this.textContent || '').length;
  return { x: 0, y: 0, width: Math.max(t * 7, 20), height: 18 };
};
dom.window.SVGElement.prototype.getComputedTextLength = function () {
  return Math.max((this.textContent || '').length * 7, 20);
};

const mermaid = (await import('mermaid')).default;
mermaid.initialize({ startOnLoad: false, theme: 'dark', securityLevel: 'loose' });
const src = fs.readFileSync('diag.mmd', 'utf8');
try {
  const { svg } = await mermaid.render('g1', src);
  fs.writeFileSync('diag.svg', svg);
  const nodes = (svg.match(/class="[^"]*\bnode\b/g) || []).length;
  const edges = (svg.match(/class="[^"]*\b(edgePath|flowchart-link)\b/g) || []).length;
  const clus = (svg.match(/class="[^"]*\bcluster\b/g) || []).length;
  console.log(`RENDER_OK  svg=${svg.length}B  nodes=${nodes}  edges=${edges}  subgraphs=${clus}`);
  const missing = ['OWL-v2', 'Kosmos-2', 'MLP', '260-dim', 'gray basket', '호출 0회']
    .filter((w) => !svg.includes(w));
  console.log(missing.length ? 'TEXT_MISSING: ' + missing.join(', ') : 'TEXT_ALL_PRESENT');
} catch (e) {
  console.log('RENDER_FAIL:', e && e.message ? e.message.split('\n').slice(0, 5).join(' | ') : String(e));
}
