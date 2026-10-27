/**
 * EVOCURE - Pancreatic Ductal Adenocarcinoma (PDAC) Tumor Microenvironment Renderer.
 * 
 * Interactive 2D Canvas renderer for cellular tumor heterogeneity, CAFs,
 * vascular supply networks, hypoxic gradient overlays, and cell-death transitions.
 */

class TumorCanvasRenderer {
  constructor(canvasElement, onSelectCellCallback) {
    this.canvas = canvasElement;
    this.ctx = this.canvas.getContext('2d');
    this.onSelectCell = onSelectCellCallback || (() => {});

    this.worldWidth = 800;
    this.worldHeight = 600;
    this.zoom = 1.0;
    this.panX = 0;
    this.panY = 0;
    this.isDragging = false;
    this.lastMouseX = 0;
    this.lastMouseY = 0;

    // Microenvironment Data Cache
    this.cells = [];
    this.vessels = [];
    this.heatmaps = {};
    this.zones = {};
    this.summaryMetrics = {};

    this.showHypoxia = true;
    this.showDrugHeatmap = true;

    this._bindEvents();
    this._startRenderLoop();
  }

  _bindEvents() {
    this.canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const rect = this.canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
      const newZoom = Math.min(Math.max(0.4, this.zoom * zoomFactor), 4.0);

      this.panX = mouseX - (mouseX - this.panX) * (newZoom / this.zoom);
      this.panY = mouseY - (mouseY - this.panY) * (newZoom / this.zoom);
      this.zoom = newZoom;
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
        const dx = e.clientX - this.lastMouseX;
        const dy = e.clientY - this.lastMouseY;
        this.panX += dx;
        this.panY += dy;
        this.lastMouseX = e.clientX;
        this.lastMouseY = e.clientY;
      }
    });

    window.addEventListener('mouseup', () => {
      this.isDragging = false;
    });

    // Select Cell
    this.canvas.addEventListener('click', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const clickX = (e.clientX - rect.left - this.panX) / this.zoom;
      const clickY = (e.clientY - rect.top - this.panY) / this.zoom;

      let nearest = null;
      let minDist = 16.0;
      for (const cell of this.cells) {
        const dist = Math.hypot(cell.x - clickX, cell.y - clickY);
        if (dist < minDist) {
          minDist = dist;
          nearest = cell;
        }
      }

      if (nearest) {
        this.onSelectCell(nearest);
      }
    });
  }

  updateState(tumorSnapshot) {
    if (!tumorSnapshot) return;
    this.cells = tumorSnapshot.cells || [];
    this.vessels = tumorSnapshot.vessels || [];
    this.heatmaps = tumorSnapshot.heatmaps || {};
    this.zones = tumorSnapshot.zones || {};
    this.summaryMetrics = tumorSnapshot.summary_metrics || {};

    if (tumorSnapshot.domain?.width) this.worldWidth = tumorSnapshot.domain.width;
    if (tumorSnapshot.domain?.height) this.worldHeight = tumorSnapshot.domain.height;
  }

  _startRenderLoop() {
    const render = () => {
      this._render();
      requestAnimationFrame(render);
    };
    requestAnimationFrame(render);
  }

  _render() {
    const ctx = this.ctx;
    const width = this.canvas.width;
    const height = this.canvas.height;

    ctx.fillStyle = '#070b16';
    ctx.fillRect(0, 0, width, height);

    ctx.save();
    ctx.translate(this.panX, this.panY);
    ctx.scale(this.zoom, this.zoom);

    // 1. Boundary
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.25)';
    ctx.lineWidth = 2;
    ctx.strokeRect(0, 0, this.worldWidth, this.worldHeight);

    // 2. Continuous Hypoxic / ECM Fields
    if (this.showHypoxia && this.heatmaps.oxygen) {
      this._renderHypoxiaOverlay(ctx);
    }

    // 3. Drug Concentration Heatmap
    if (this.showDrugHeatmap && this.heatmaps.drug_free) {
      this._renderDrugHeatmap(ctx);
    }

    // 4. Blood Vessels Network
    this._renderVessels(ctx);

    // 5. Individual Cellular Agents
    this._renderCells(ctx);

    ctx.restore();
  }

  _renderHypoxiaOverlay(ctx) {
    const o2Grid = this.heatmaps.oxygen;
    const ny = o2Grid.length;
    const nx = o2Grid[0].length;
    const cellW = this.worldWidth / nx;
    const cellH = this.worldHeight / ny;

    ctx.save();
    for (let r = 0; r < ny; r++) {
      for (let c = 0; c < nx; c++) {
        const o2 = o2Grid[r][c];
        if (o2 < 0.30) {
          // Hypoxic mantle or necrotic core
          const alpha = Math.min(0.35, (0.30 - o2) / 0.30);
          ctx.fillStyle = `rgba(100, 116, 139, ${alpha.toFixed(3)})`;
          ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
        }
      }
    }
    ctx.restore();
  }

  _renderDrugHeatmap(ctx) {
    const drugGrid = this.heatmaps.drug_free;
    if (!drugGrid || !drugGrid.length) return;
    const ny = drugGrid.length;
    const nx = drugGrid[0].length;
    const cellW = this.worldWidth / nx;
    const cellH = this.worldHeight / ny;

    ctx.save();
    for (let r = 0; r < ny; r++) {
      for (let c = 0; c < nx; c++) {
        const conc = drugGrid[r][c];
        if (conc > 0.05) {
          const alpha = Math.min(0.35, conc / 5.0);
          ctx.fillStyle = `rgba(56, 189, 248, ${alpha.toFixed(3)})`;
          ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
        }
      }
    }
    ctx.restore();
  }

  _renderVessels(ctx) {
    ctx.save();
    ctx.strokeStyle = 'rgba(168, 85, 247, 0.4)';
    ctx.lineWidth = 4;
    for (const v of this.vessels) {
      ctx.beginPath();
      ctx.arc(v.x, v.y, v.radius || 8, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(168, 85, 247, 0.2)';
      ctx.fill();
      ctx.stroke();
    }
    ctx.restore();
  }

  _renderCells(ctx) {
    for (const cell of this.cells) {
      const radius = cell.r || 6.5;
      const type = cell.t || cell.cell_type;
      const state = cell.st || cell.death_mode;

      ctx.save();

      // Color and shape coding
      let col = '#ef4444'; // Sensitive
      let shape = 'circle';

      if (state === 'APOPTOSIS') {
        col = '#a855f7';
        shape = 'star';
      } else if (state === 'NECROSIS') {
        col = '#475569';
        shape = 'cross';
      } else if (type === 'TUMOR_RESISTANT') {
        col = '#f97316';
        shape = 'hexagon';
      } else if (type === 'EPITHELIAL_HEALTHY') {
        col = '#10b981';
        shape = 'square';
      } else if (type === 'CAF' || type === 'FIBROBLAST') {
        col = '#eab308';
        shape = 'triangle';
      } else if (type === 'IMMUNE_T') {
        col = '#3b82f6';
        shape = 'diamond';
      } else if (type === 'MACROPHAGE') {
        col = '#06b6d4';
        shape = 'diamond';
      } else if (type === 'ENDOTHELIAL') {
        col = '#8b5cf6';
        shape = 'ring';
      }

      ctx.fillStyle = col;
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.4)';
      ctx.lineWidth = 1;

      ctx.beginPath();
      if (shape === 'circle') {
        ctx.arc(cell.x, cell.y, radius, 0, Math.PI * 2);
      } else if (shape === 'hexagon') {
        this._drawPoly(ctx, cell.x, cell.y, radius, 6);
      } else if (shape === 'square') {
        ctx.rect(cell.x - radius, cell.y - radius, radius * 2, radius * 2);
      } else if (shape === 'triangle') {
        this._drawPoly(ctx, cell.x, cell.y, radius, 3);
      } else if (shape === 'diamond') {
        this._drawPoly(ctx, cell.x, cell.y, radius, 4);
      } else if (shape === 'star') {
        this._drawStar(ctx, cell.x, cell.y, 5, radius, radius * 0.45);
      } else if (shape === 'cross') {
        ctx.arc(cell.x, cell.y, radius * 0.7, 0, Math.PI * 2);
      } else {
        ctx.arc(cell.x, cell.y, radius, 0, Math.PI * 2);
      }

      ctx.fill();
      ctx.stroke();

      // Stress ring
      if (cell.v < 0.6 && state === 'ALIVE') {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 2]);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      ctx.restore();
    }
  }

  _drawPoly(ctx, cx, cy, radius, sides) {
    const angleStep = (Math.PI * 2) / sides;
    ctx.moveTo(cx + radius * Math.cos(0), cy + radius * Math.sin(0));
    for (let i = 1; i <= sides; i++) {
      ctx.lineTo(cx + radius * Math.cos(i * angleStep), cy + radius * Math.sin(i * angleStep));
    }
  }

  _drawStar(ctx, cx, cy, spikes, outerRadius, innerRadius) {
    let rot = (Math.PI / 2) * 3;
    let step = Math.PI / spikes;
    ctx.moveTo(cx, cy - outerRadius);
    for (let i = 0; i < spikes; i++) {
      ctx.lineTo(cx + Math.cos(rot) * outerRadius, cy + Math.sin(rot) * outerRadius);
      rot += step;
      ctx.lineTo(cx + Math.cos(rot) * innerRadius, cy + Math.sin(rot) * innerRadius);
      rot += step;
    }
    ctx.lineTo(cx, cy - outerRadius);
  }
}

window.TumorCanvasRenderer = TumorCanvasRenderer;
