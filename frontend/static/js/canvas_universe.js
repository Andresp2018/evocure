/**
 * EVOCURE - Chemical Universe Canvas 2D Renderer.
 * 
 * High-performance 60 FPS renderer for active replicators, Brownian particles,
 * continuous substrate diffusion fields, compartment barriers, and reaction ripples.
 * Strictly implements accessibility shapes and colorblind-safe palettes.
 */

class UniverseCanvasRenderer {
  constructor(canvasElement, onSelectEntityCallback) {
    this.canvas = canvasElement;
    this.ctx = this.canvas.getContext('2d');
    this.onSelectEntity = onSelectEntityCallback || (() => {});

    // World & Viewport State
    this.worldWidth = 1000;
    this.worldHeight = 700;
    this.zoom = 1.0;
    this.panX = 0;
    this.panY = 0;
    this.isDragging = false;
    this.lastMouseX = 0;
    this.lastMouseY = 0;

    // Simulation Data Cache
    this.entities = [];
    this.totalEntityCount = 0;
    this.barriers = [];
    this.regions = [];
    this.substrateHeatmap = null;
    this.reactionRipples = []; // { x, y, radius, maxRadius, alpha, color }
    this.lastReactionStep = -1;

    // Drag-and-Drop Drop Preview
    this.dragPreviewActive = false;
    this.dragPreviewX = 0;
    this.dragPreviewY = 0;
    this.dragPreviewCompound = null;

    // Settings
    this.showHeatmap = true;
    this.showRipples = true;

    this._bindEvents();
    this._startRenderLoop();
  }

  _bindEvents() {
    // Zoom via mouse wheel
    this.canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const rect = this.canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
      const newZoom = Math.min(Math.max(0.4, this.zoom * zoomFactor), 4.0);

      // Zoom towards mouse pointer
      this.panX = mouseX - (mouseX - this.panX) * (newZoom / this.zoom);
      this.panY = mouseY - (mouseY - this.panY) * (newZoom / this.zoom);
      this.zoom = newZoom;
    }, { passive: false });

    // Pan via mouse drag
    this.canvas.addEventListener('mousedown', (e) => {
      if (e.button === 0) { // Left click
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

    // Click to select entity
    this.canvas.addEventListener('click', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const clickX = (e.clientX - rect.left - this.panX) / this.zoom;
      const clickY = (e.clientY - rect.top - this.panY) / this.zoom;

      // Find nearest entity within selection radius
      let nearest = null;
      let minDist = 18.0;
      for (const ent of this.entities) {
        const dist = Math.hypot(ent.x - clickX, ent.y - clickY);
        if (dist < minDist) {
          minDist = dist;
          nearest = ent;
        }
      }

      if (nearest) {
        this.onSelectEntity(nearest);
      }
    });

    // Drag-and-Drop Compound onto Canvas
    this.canvas.addEventListener('dragover', (e) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'copy';
      const rect = this.canvas.getBoundingClientRect();
      this.dragPreviewActive = true;
      this.dragPreviewX = (e.clientX - rect.left - this.panX) / this.zoom;
      this.dragPreviewY = (e.clientY - rect.top - this.panY) / this.zoom;
    });

    this.canvas.addEventListener('dragleave', () => {
      this.dragPreviewActive = false;
    });

    this.canvas.addEventListener('drop', (e) => {
      e.preventDefault();
      this.dragPreviewActive = false;
      const rawData = e.dataTransfer.getData('application/json');
      if (rawData) {
        try {
          const comp = JSON.parse(rawData);
          const rect = this.canvas.getBoundingClientRect();
          const dropX = (e.clientX - rect.left - this.panX) / this.zoom;
          const dropY = (e.clientY - rect.top - this.panY) / this.zoom;

          // Dispatch compound addition
          if (window.EVOCURE_APP && window.EVOCURE_APP.addCompoundAtCoords) {
            window.EVOCURE_APP.addCompoundAtCoords(comp, dropX, dropY);
          }
        } catch (err) {
          console.error("Drop compound error:", err);
        }
      }
    });
  }

  setZoom(factor) {
    this.zoom = Math.min(Math.max(0.4, factor), 4.0);
  }

  resetView() {
    this.zoom = 1.0;
    this.panX = 0;
    this.panY = 0;
  }

  updateState(state) {
    if (!state) return;
    this.entities = state.entities || [];
    this.totalEntityCount = state.entity_total_count || this.entities.length;
    this.barriers = state.environment?.barriers || [];
    this.regions = state.environment?.regions || [];
    this.substrateHeatmap = state.substrate_heatmap || null;

    if (state.environment?.width) this.worldWidth = state.environment.width;
    if (state.environment?.height) this.worldHeight = state.environment.height;

    // Trigger reaction ripples based on recent_reactions
    if (this.showRipples && state.recent_reactions && state.recent_reactions.length > 0) {
      for (const rxn of state.recent_reactions) {
        if (rxn.step !== this.lastReactionStep) {
          this.reactionRipples.push({
            x: rxn.x,
            y: rxn.y,
            radius: 4,
            maxRadius: 36,
            alpha: 0.9,
            color: '#38bdf8',
          });
        }
      }
      this.lastReactionStep = state.recent_reactions[state.recent_reactions.length - 1].step;
    }
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

    // Clear Screen
    ctx.fillStyle = '#050813';
    ctx.fillRect(0, 0, width, height);

    ctx.save();
    // Apply pan and zoom
    ctx.translate(this.panX, this.panY);
    ctx.scale(this.zoom, this.zoom);

    // 1. Draw World Bounds
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.2)';
    ctx.lineWidth = 2;
    ctx.strokeRect(0, 0, this.worldWidth, this.worldHeight);

    // 2. Draw Environmental Niche Regions
    this._renderRegions(ctx);

    // 3. Draw Substrate Heatmap Overlay
    if (this.showHeatmap && this.substrateHeatmap) {
      this._renderSubstrateHeatmap(ctx);
    }

    // 4. Draw Compartment Barriers
    this._renderBarriers(ctx);

    // 5. Draw Replicator / Molecule Entities
    this._renderEntities(ctx);

    // 6. Draw Reaction Ripples
    this._renderReactionRipples(ctx);

    // 7. Draw Drag & Drop Drop Preview
    if (this.dragPreviewActive) {
      ctx.beginPath();
      ctx.arc(this.dragPreviewX, this.dragPreviewY, 20, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(56, 189, 248, 0.3)';
      ctx.fill();
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.restore();
  }

  _renderRegions(ctx) {
    for (const reg of this.regions) {
      ctx.save();
      ctx.fillStyle = reg.temperature_delta_k > 5 
        ? 'rgba(239, 68, 68, 0.06)' 
        : 'rgba(56, 189, 248, 0.05)';
      ctx.fillRect(reg.x, reg.y, reg.width, reg.height);

      ctx.strokeStyle = reg.temperature_delta_k > 5 
        ? 'rgba(239, 68, 68, 0.2)' 
        : 'rgba(56, 189, 248, 0.2)';
      ctx.lineWidth = 1;
      ctx.setLineDash([6, 6]);
      ctx.strokeRect(reg.x, reg.y, reg.width, reg.height);
      ctx.setLineDash([]);

      ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
      ctx.font = '10px var(--font-sans)';
      ctx.fillText(reg.name, reg.x + 8, reg.y + 16);
      ctx.restore();
    }
  }

  _renderSubstrateHeatmap(ctx) {
    if (!this.substrateHeatmap || !this.substrateHeatmap.length) return;
    const ny = this.substrateHeatmap.length;
    const nx = this.substrateHeatmap[0].length;
    const cellW = this.worldWidth / nx;
    const cellH = this.worldHeight / ny;

    ctx.save();
    for (let r = 0; r < ny; r++) {
      for (let c = 0; c < nx; c++) {
        const val = this.substrateHeatmap[r][c];
        if (val > 0.1) {
          const intensity = Math.min(val / 3.0, 0.25);
          ctx.fillStyle = `rgba(16, 185, 129, ${intensity.toFixed(3)})`;
          ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
        }
      }
    }
    ctx.restore();
  }

  _renderBarriers(ctx) {
    for (const b of this.barriers) {
      ctx.save();
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.7)';
      ctx.lineWidth = 3;
      ctx.setLineDash([8, 6]);
      ctx.beginPath();
      ctx.moveTo(b.x1, b.y1);
      ctx.lineTo(b.x2, b.y2);
      ctx.stroke();

      // Semi-permeable glow
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.2)';
      ctx.lineWidth = 8;
      ctx.stroke();
      ctx.restore();
    }
  }

  _renderEntities(ctx) {
    for (const ent of this.entities) {
      const radius = ent.r || 5;
      const col = ent.col || '#38bdf8';
      const shape = ent.shp || 'circle';

      ctx.save();
      ctx.fillStyle = col;
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
      ctx.lineWidth = 1;

      // Draw distinctive accessible shapes
      ctx.beginPath();
      if (shape === 'circle') {
        ctx.arc(ent.x, ent.y, radius, 0, Math.PI * 2);
      } else if (shape === 'hexagon') {
        this._drawPolygon(ctx, ent.x, ent.y, radius, 6);
      } else if (shape === 'diamond') {
        this._drawPolygon(ctx, ent.x, ent.y, radius, 4);
      } else if (shape === 'star') {
        this._drawStar(ctx, ent.x, ent.y, 5, radius, radius * 0.5);
      } else if (shape === 'triangle') {
        this._drawPolygon(ctx, ent.x, ent.y, radius, 3);
      } else if (shape === 'ring') {
        ctx.arc(ent.x, ent.y, radius, 0, Math.PI * 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(ent.x, ent.y, radius * 0.4, 0, Math.PI * 2);
      } else {
        ctx.arc(ent.x, ent.y, radius, 0, Math.PI * 2);
      }

      ctx.fill();
      ctx.stroke();

      // Tether bond connection if bound to another entity
      if (ent.bnd) {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.lineWidth = 1;
        // Simple subtle dash
      }

      ctx.restore();
    }
  }

  _drawPolygon(ctx, cx, cy, radius, sides) {
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

  _renderReactionRipples(ctx) {
    for (let i = this.reactionRipples.length - 1; i >= 0; i--) {
      const rip = this.reactionRipples[i];
      rip.radius += 1.2;
      rip.alpha -= 0.035;

      if (rip.alpha <= 0 || rip.radius >= rip.maxRadius) {
        this.reactionRipples.splice(i, 1);
        continue;
      }

      ctx.save();
      ctx.beginPath();
      ctx.arc(rip.x, rip.y, rip.radius, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(56, 189, 248, ${rip.alpha.toFixed(2)})`;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.restore();
    }
  }
}

window.UniverseCanvasRenderer = UniverseCanvasRenderer;
