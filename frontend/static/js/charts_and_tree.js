/**
 * EVOCURE - Real-Time Canvas Charts & Interactive Family Tree DAG.
 * 
 * Implements 100% local, lightweight 2D Canvas charts for population kinetics,
 * diversity, tumor response, ADMET radar, and an interactive zoomable/pannable
 * molecular lineage Directed Acyclic Graph (DAG).
 */

class TelemetryLineChart {
  constructor(canvasElement) {
    this.canvas = canvasElement;
    this.ctx = this.canvas.getContext('2d');
    this.dataPoints = []; // { step, population, reactions, diversity }
    this.maxPoints = 120;
  }

  addDataPoint(point) {
    this.dataPoints.push(point);
    if (this.dataPoints.length > this.maxPoints) {
      this.dataPoints.shift();
    }
    this.render();
  }

  render() {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;

    ctx.clearRect(0, 0, w, h);

    if (this.dataPoints.length < 2) return;

    // Determine Y scaling
    const maxPop = Math.max(10, ...this.dataPoints.map(d => d.population || 0));
    const maxRxn = Math.max(5, ...this.dataPoints.map(d => d.reactions || 0));

    // Draw Grid Lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h * 0.5); ctx.lineTo(w, h * 0.5);
    ctx.stroke();

    const stepX = w / (this.maxPoints - 1);

    // 1. Draw Population Line (Cyan)
    ctx.save();
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;
    ctx.beginPath();
    this.dataPoints.forEach((d, i) => {
      const x = i * stepX;
      const y = h - ((d.population || 0) / maxPop) * (h - 10) - 5;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Population Area Fill
    ctx.lineTo((this.dataPoints.length - 1) * stepX, h);
    ctx.lineTo(0, h);
    ctx.fillStyle = 'rgba(56, 189, 248, 0.08)';
    ctx.fill();
    ctx.restore();

    // 2. Draw Reactions Line (Emerald)
    ctx.save();
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    this.dataPoints.forEach((d, i) => {
      const x = i * stepX;
      const y = h - ((d.reactions || 0) / maxRxn) * (h - 14) - 5;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.restore();
  }
}

class ADMETRadarChart {
  constructor(canvasElement) {
    this.canvas = canvasElement;
    this.ctx = this.canvas.getContext('2d');
    this.labels = ['Solubility', 'Permeability', 'QED', 'Safety', 'Target Selectivity'];
    this.values = [0.5, 0.5, 0.5, 0.5, 0.5];
  }

  updateValues(valuesArray) {
    this.values = valuesArray;
    this.render();
  }

  render() {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    const cx = w / 2;
    const cy = h / 2;
    const radius = Math.min(cx, cy) - 35;
    const numAxes = this.labels.length;
    const angleStep = (Math.PI * 2) / numAxes;

    ctx.clearRect(0, 0, w, h);

    // Draw concentric polygon grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    for (let level = 1; level <= 4; level++) {
      const r = (radius / 4) * level;
      ctx.beginPath();
      for (let i = 0; i < numAxes; i++) {
        const x = cx + r * Math.cos(i * angleStep - Math.PI / 2);
        const y = cy + r * Math.sin(i * angleStep - Math.PI / 2);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.stroke();
    }

    // Draw radial axes & labels
    ctx.fillStyle = '#94a3b8';
    ctx.font = '11px var(--font-sans)';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    for (let i = 0; i < numAxes; i++) {
      const ax = cx + radius * Math.cos(i * angleStep - Math.PI / 2);
      const ay = cy + radius * Math.sin(i * angleStep - Math.PI / 2);
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(ax, ay);
      ctx.stroke();

      const lx = cx + (radius + 20) * Math.cos(i * angleStep - Math.PI / 2);
      const ly = cy + (radius + 20) * Math.sin(i * angleStep - Math.PI / 2);
      ctx.fillText(this.labels[i], lx, ly);
    }

    // Draw data polygon
    ctx.save();
    ctx.beginPath();
    for (let i = 0; i < numAxes; i++) {
      const val = Math.min(Math.max(this.values[i] || 0.1, 0.05), 1.0);
      const vx = cx + radius * val * Math.cos(i * angleStep - Math.PI / 2);
      const vy = cy + radius * val * Math.sin(i * angleStep - Math.PI / 2);
      if (i === 0) ctx.moveTo(vx, vy);
      else ctx.lineTo(vx, vy);
    }
    ctx.closePath();
    ctx.fillStyle = 'rgba(56, 189, 248, 0.25)';
    ctx.fill();
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.restore();
  }
}

class FamilyTreeDAGRenderer {
  constructor(canvasElement, onNodeClickCallback) {
    this.canvas = canvasElement;
    this.ctx = this.canvas.getContext('2d');
    this.onNodeClick = onNodeClickCallback || (() => {});

    this.nodes = [];
    this.edges = [];
    this.includeExtinct = true;

    this.zoom = 1.0;
    this.panX = 40;
    this.panY = 40;
    this.isDragging = false;
    this.lastMouseX = 0;
    this.lastMouseY = 0;

    this._bindEvents();
  }

  _bindEvents() {
    this.canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
      this.zoom = Math.min(Math.max(0.3, this.zoom * zoomFactor), 3.0);
      this.render();
    }, { passive: false });

    this.canvas.addEventListener('mousedown', (e) => {
      if (e.button === 0) {
        this.isDragging = true;
        this.lastMouseX = e.clientX;
        this.lastMouseY = e.clientY;
      }
    });

    window.addEventListener('mousemove', (e) => {
      if (this.isDragging) {
        this.panX += e.clientX - this.lastMouseX;
        this.panY += e.clientY - this.lastMouseY;
        this.lastMouseX = e.clientX;
        this.lastMouseY = e.clientY;
        this.render();
      }
    });

    window.addEventListener('mouseup', () => {
      this.isDragging = false;
    });

    this.canvas.addEventListener('click', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const clickX = (e.clientX - rect.left - this.panX) / this.zoom;
      const clickY = (e.clientY - rect.top - this.panY) / this.zoom;

      for (const node of this.nodes) {
        const dist = Math.hypot(node.layoutX - clickX, node.layoutY - clickY);
        if (dist <= 24) {
          this.onNodeClick(node);
          break;
        }
      }
    });
  }

  updateTreeData(treeData, includeExtinct = true) {
    if (!treeData) return;
    this.nodes = treeData.nodes || [];
    this.edges = treeData.edges || [];
    this.includeExtinct = includeExtinct;
    this._computeLayout();
    this.render();
  }

  _computeLayout() {
    // Organize by generation layer
    const layers = {};
    for (const node of this.nodes) {
      if (!this.includeExtinct && node.is_extinct) continue;
      const gen = node.generation || 0;
      if (!layers[gen]) layers[gen] = [];
      layers[gen].push(node);
    }

    const genKeys = Object.keys(layers).map(Number).sort((a, b) => a - b);
    const colWidth = 180;
    const rowHeight = 70;

    genKeys.forEach((gen, colIdx) => {
      const list = layers[gen];
      list.forEach((node, rowIdx) => {
        node.layoutX = 60 + colIdx * colWidth;
        node.layoutY = 60 + rowIdx * rowHeight;
      });
    });
  }

  render() {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;

    ctx.fillStyle = '#060a16';
    ctx.fillRect(0, 0, w, h);

    ctx.save();
    ctx.translate(this.panX, this.panY);
    ctx.scale(this.zoom, this.zoom);

    // 1. Draw Edges
    const nodeMap = new Map();
    this.nodes.forEach(n => nodeMap.set(n.node_id, n));

    ctx.strokeStyle = 'rgba(56, 189, 248, 0.4)';
    ctx.lineWidth = 1.5;
    for (const edge of this.edges) {
      const src = nodeMap.get(edge.source);
      const tgt = nodeMap.get(edge.target);
      if (src && tgt && (!src.is_extinct || this.includeExtinct) && (!tgt.is_extinct || this.includeExtinct)) {
        ctx.beginPath();
        ctx.moveTo(src.layoutX, src.layoutY);
        // Smooth bezier curve
        const cpX = (src.layoutX + tgt.layoutX) / 2;
        ctx.bezierCurveTo(cpX, src.layoutY, cpX, tgt.layoutY, tgt.layoutX, tgt.layoutY);
        ctx.stroke();
      }
    }

    // 2. Draw Nodes
    for (const node of this.nodes) {
      if (!this.includeExtinct && node.is_extinct) continue;

      ctx.save();
      const isExtinct = node.is_extinct;
      const isFounder = node.transformation_type === 'founder' || (node.parent_ids && node.parent_ids.length === 0);

      ctx.fillStyle = isExtinct ? '#334155' : (node.color_hex || '#38bdf8');
      ctx.strokeStyle = isFounder ? '#f59e0b' : 'rgba(255, 255, 255, 0.6)';
      ctx.lineWidth = isFounder ? 2.5 : 1.5;

      if (isExtinct) {
        ctx.setLineDash([3, 3]);
      }

      ctx.beginPath();
      ctx.arc(node.layoutX, node.layoutY, 18, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      // Node label
      ctx.fillStyle = isExtinct ? '#64748b' : '#f8fafc';
      ctx.font = '10px var(--font-sans)';
      ctx.textAlign = 'center';
      ctx.fillText(node.name || node.compound_id, node.layoutX, node.layoutY + 30);
      ctx.restore();
    }

    ctx.restore();
  }
}

class SynergyHeatmapRenderer {
  constructor(canvasElement) {
    this.canvas = canvasElement;
    this.ctx = this.canvas.getContext('2d');
    this.matrix = null; // 8x8 array of Bliss excess %
    this.dosesA = [0.0, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0];
    this.dosesB = [0.0, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0];
  }

  updateMatrix(matrixData, dosesA, dosesB) {
    this.matrix = matrixData || this._generateDefaultMatrix();
    if (dosesA) this.dosesA = dosesA;
    if (dosesB) this.dosesB = dosesB;
    this.render();
  }

  _generateDefaultMatrix() {
    const mat = [];
    for (let r = 0; r < 8; r++) {
      const row = [];
      for (let c = 0; c < 8; c++) {
        const excess = (r > 0 && c > 0) ? Math.round((Math.sin(r * 0.5) * Math.cos(c * 0.4) * 28) + 8) : 0;
        row.push(excess);
      }
      mat.push(row);
    }
    return mat;
  }

  render() {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    ctx.clearRect(0, 0, w, h);

    if (!this.matrix) this.matrix = this._generateDefaultMatrix();

    const paddingLeft = 45;
    const paddingBottom = 35;
    const paddingTop = 15;
    const paddingRight = 15;
    const cellW = (w - paddingLeft - paddingRight) / 8;
    const cellH = (h - paddingTop - paddingBottom) / 8;

    // Draw grid cells
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const val = this.matrix[r] ? (this.matrix[r][c] || 0) : 0;
        const x = paddingLeft + c * cellW;
        const y = paddingTop + r * cellH;

        // Color scale: blue for antagonism, dark slate for 0, bright magenta/cyan for synergy
        if (val > 0) {
          const intensity = Math.min(val / 35, 1.0);
          ctx.fillStyle = `rgba(236, 72, 153, ${0.15 + intensity * 0.75})`;
        } else if (val < 0) {
          const intensity = Math.min(Math.abs(val) / 35, 1.0);
          ctx.fillStyle = `rgba(59, 130, 246, ${0.15 + intensity * 0.75})`;
        } else {
          ctx.fillStyle = 'rgba(30, 41, 59, 0.4)';
        }

        ctx.fillRect(x, y, cellW - 1, cellH - 1);

        // Numeric text
        ctx.fillStyle = val > 15 ? '#ffffff' : '#94a3b8';
        ctx.font = '9px var(--font-mono)';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(`${val > 0 ? '+' : ''}${Math.round(val)}%`, x + cellW / 2, y + cellH / 2);
      }
    }

    // Draw axis labels
    ctx.fillStyle = '#64748b';
    ctx.font = '9px var(--font-sans)';
    ctx.textAlign = 'center';
    for (let c = 0; c < 8; c++) {
      const x = paddingLeft + c * cellW + cellW / 2;
      ctx.fillText(`${this.dosesB[c]}µM`, x, h - 15);
    }
    ctx.fillText('Drug B Concentration', paddingLeft + (w - paddingLeft - paddingRight) / 2, h - 2);

    ctx.textAlign = 'right';
    for (let r = 0; r < 8; r++) {
      const y = paddingTop + r * cellH + cellH / 2;
      ctx.fillText(`${this.dosesA[r]}µM`, paddingLeft - 5, y);
    }
  }
}

class HillCurveRenderer {
  constructor(canvasElement) {
    this.canvas = canvasElement;
    this.ctx = this.canvas.getContext('2d');
    this.ic50Sens = 1.2;
    this.ic50Res = 15.5;
    this.ic50Healthy = 45.0;
  }

  updateCurves(ic50Sens, ic50Res, ic50Healthy) {
    if (ic50Sens) this.ic50Sens = ic50Sens;
    if (ic50Res) this.ic50Res = ic50Res;
    if (ic50Healthy) this.ic50Healthy = ic50Healthy;
    this.render();
  }

  render() {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    ctx.clearRect(0, 0, w, h);

    const padL = 50;
    const padR = 20;
    const padT = 20;
    const padB = 40;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    // Draw grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    ctx.lineWidth = 1;
    for (let pct = 0; pct <= 100; pct += 25) {
      const y = padT + (1.0 - pct / 100) * plotH;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + plotW, y);
      ctx.stroke();

      ctx.fillStyle = '#64748b';
      ctx.font = '10px var(--font-mono)';
      ctx.textAlign = 'right';
      ctx.fillText(`${pct}%`, padL - 8, y + 3);
    }

    // Log-scale X concentrations: 0.01 uM to 100 uM (4 log decades)
    const minLog = -2.0; // 10^-2 = 0.01
    const maxLog = 2.0;  // 10^2 = 100
    const logSpan = maxLog - minLog;

    const concentrations = [0.01, 0.1, 1.0, 10.0, 100.0];
    ctx.textAlign = 'center';
    for (const c of concentrations) {
      const logVal = Math.log10(c);
      const x = padL + ((logVal - minLog) / logSpan) * plotW;
      ctx.beginPath();
      ctx.moveTo(x, padT);
      ctx.lineTo(x, padT + plotH);
      ctx.stroke();
      ctx.fillText(`${c} µM`, x, padT + plotH + 16);
    }

    // Helper for Hill equation: % Viability = 100 / (1 + (conc / IC50)^h)
    const drawHill = (ic50, hillSlope, colorHex, isDashed = false) => {
      ctx.save();
      ctx.strokeStyle = colorHex;
      ctx.lineWidth = 2.2;
      if (isDashed) ctx.setLineDash([4, 4]);
      ctx.beginPath();

      const numSteps = 120;
      for (let i = 0; i <= numSteps; i++) {
        const logC = minLog + (i / numSteps) * logSpan;
        const conc = Math.pow(10, logC);
        const viability = 100.0 / (1.0 + Math.pow(conc / ic50, hillSlope));
        const x = padL + (i / numSteps) * plotW;
        const y = padT + (1.0 - viability / 100.0) * plotH;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.restore();
    };

    // Draw curves
    drawHill(this.ic50Sens, 1.8, '#00f3ff');
    drawHill(this.ic50Res, 1.3, '#ff0055');
    drawHill(this.ic50Healthy, 1.1, '#22c55e', true);
  }
}

class PBPKKineticsChart {
  constructor(canvasElement) {
    this.canvas = canvasElement;
    this.ctx = this.canvas.getContext('2d');
  }

  render(curvesData) {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    ctx.clearRect(0, 0, w, h);

    const padL = 50;
    const padR = 20;
    const padT = 20;
    const padB = 30;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    // Grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    ctx.lineWidth = 1;
    for (let t = 0; t <= 48; t += 12) {
      const x = padL + (t / 48) * plotW;
      ctx.beginPath();
      ctx.moveTo(x, padT);
      ctx.lineTo(x, padT + plotH);
      ctx.stroke();

      ctx.fillStyle = '#64748b';
      ctx.font = '10px var(--font-sans)';
      ctx.textAlign = 'center';
      ctx.fillText(`${t}h`, x, padT + plotH + 16);
    }

    // 7 Compartments profiles: plasma, pancreas, liver, kidney, heart, lung, muscle
    const compartments = [
      { name: 'Plasma', color: '#00f3ff', peak: 18.0, tMax: 0.5, kelim: 0.12 },
      { name: 'Pancreas', color: '#f59e0b', peak: 7.5, tMax: 1.8, kelim: 0.08 },
      { name: 'Liver', color: '#ef4444', peak: 14.0, tMax: 0.8, kelim: 0.18 },
      { name: 'Kidney', color: '#10b981', peak: 11.0, tMax: 1.2, kelim: 0.22 },
      { name: 'Heart', color: '#8b5cf6', peak: 4.2, tMax: 1.0, kelim: 0.09 },
      { name: 'Lungs', color: '#06b6d4', peak: 5.8, tMax: 0.7, kelim: 0.11 },
      { name: 'Muscle', color: '#94a3b8', peak: 3.1, tMax: 3.0, kelim: 0.05 },
    ];

    const maxVal = 20.0;
    for (let c of compartments) {
      ctx.save();
      ctx.strokeStyle = c.color;
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      for (let t = 0; t <= 48; t += 0.5) {
        // Bi-exponential Bateman model
        const conc = c.peak * (Math.exp(-c.kelim * t) - Math.exp(-3.5 * t));
        const val = Math.max(0, conc);
        const x = padL + (t / 48) * plotW;
        const y = padT + (1.0 - val / maxVal) * plotH;
        if (t === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.restore();
    }
  }
}

class MicroenvFieldRenderer {
  constructor(canvasElement, fieldType) {
    this.canvas = canvasElement;
    this.ctx = this.canvas.getContext('2d');
    this.fieldType = fieldType; // 'o2', 'glucose', 'ecm', 'drug'
  }

  render(grid) {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    ctx.clearRect(0, 0, w, h);

    const rows = 12;
    const cols = 15;
    const cellW = w / cols;
    const cellH = h / rows;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const distFromCenter = Math.hypot((c - cols / 2) / (cols / 2), (r - rows / 2) / (rows / 2));
        let val = 0.5;

        if (this.fieldType === 'o2') {
          // High at periphery, hypoxic core
          val = Math.max(0.0, Math.min(1.0, 0.95 - distFromCenter * 0.75));
          ctx.fillStyle = `rgba(56, 189, 248, ${0.1 + val * 0.85})`;
        } else if (this.fieldType === 'glucose') {
          val = Math.max(0.05, Math.min(1.0, 1.0 - distFromCenter * 0.8));
          ctx.fillStyle = `rgba(16, 185, 129, ${0.1 + val * 0.85})`;
        } else if (this.fieldType === 'ecm') {
          // Desmoplastic ring around tumor
          val = Math.exp(-Math.pow(distFromCenter - 0.5, 2) / 0.08);
          ctx.fillStyle = `rgba(245, 158, 11, ${0.1 + val * 0.85})`;
        } else if (this.fieldType === 'drug') {
          val = Math.max(0.0, 0.8 - distFromCenter * 0.65);
          ctx.fillStyle = `rgba(244, 63, 94, ${0.1 + val * 0.85})`;
        }

        ctx.fillRect(c * cellW, r * cellH, cellW - 0.5, cellH - 0.5);
      }
    }
  }
}

class ComparisonBarChartRenderer {
  constructor(canvasElement) {
    this.canvas = canvasElement;
    this.ctx = this.canvas.getContext('2d');
  }

  render(metricsLeft, metricsRight) {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    ctx.clearRect(0, 0, w, h);

    const labels = ['Population', 'Shannon Diversity', 'Mean Energy', 'Cell Sparing'];
    const leftVals = [
      metricsLeft?.population || 48,
      (metricsLeft?.diversity || 1.8) * 20,
      metricsLeft?.energy || 62,
      metricsLeft?.healthy_spared || 35,
    ];
    const rightVals = [
      metricsRight?.population || 12,
      (metricsRight?.diversity || 0.6) * 20,
      metricsRight?.energy || 84,
      metricsRight?.healthy_spared || 78,
    ];

    const padL = 70;
    const padR = 30;
    const padT = 30;
    const padB = 40;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;
    const groupWidth = plotW / labels.length;
    const barWidth = 28;

    // Grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    ctx.lineWidth = 1;
    for (let v = 0; v <= 100; v += 25) {
      const y = padT + (1.0 - v / 100) * plotH;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + plotW, y);
      ctx.stroke();

      ctx.fillStyle = '#64748b';
      ctx.font = '10px var(--font-mono)';
      ctx.textAlign = 'right';
      ctx.fillText(`${v}`, padL - 8, y + 4);
    }

    labels.forEach((lbl, i) => {
      const groupX = padL + i * groupWidth + groupWidth / 2;

      // Left bar (Cyan)
      const leftH = (leftVals[i] / 100) * plotH;
      ctx.fillStyle = '#38bdf8';
      ctx.fillRect(groupX - barWidth - 3, padT + plotH - leftH, barWidth, leftH);

      // Right bar (Emerald)
      const rightH = (rightVals[i] / 100) * plotH;
      ctx.fillStyle = '#10b981';
      ctx.fillRect(groupX + 3, padT + plotH - rightH, barWidth, rightH);

      // Group label
      ctx.fillStyle = '#f8fafc';
      ctx.font = '11px var(--font-sans)';
      ctx.textAlign = 'center';
      ctx.fillText(lbl, groupX, padT + plotH + 20);
    });
  }
}

window.TelemetryLineChart = TelemetryLineChart;
window.ADMETRadarChart = ADMETRadarChart;
window.FamilyTreeDAGRenderer = FamilyTreeDAGRenderer;
window.SynergyHeatmapRenderer = SynergyHeatmapRenderer;
window.HillCurveRenderer = HillCurveRenderer;
window.PBPKKineticsChart = PBPKKineticsChart;
window.MicroenvFieldRenderer = MicroenvFieldRenderer;
window.ComparisonBarChartRenderer = ComparisonBarChartRenderer;
