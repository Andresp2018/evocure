/**
 * EVOCURE PLAY Mode Controller.
 * 
 * Manages the simplified, high-energy gaming & learning experience:
 * - 4-Card Quick-Launch Hub (< 3 clicks to start)
 * - Game-like tool palette & periodic milestone modals
 * - Simplified Cancer Battle and Compound Cards
 * - 8 Educational Missions, Achievements, Glossary, and 10-Step Tutorial
 */

class PlayModeManager {
  constructor(appController) {
    this.app = appController;
    this.currentTutorialStep = 1;
    this.totalTutorialSteps = 10;
    this.tutorialStepsData = [];
    this.missionsData = [];
    this.glossaryData = [];

    this._initDomListeners();
  }

  _initDomListeners() {
    // 4 Quick-Launch Cards
    document.querySelectorAll('.launch-card').forEach(card => {
      card.addEventListener('click', () => {
        const preset = card.dataset.preset;
        this.launchPreset(preset);
      });
      card.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          this.launchPreset(card.dataset.preset);
        }
      });
    });

    // Surprise Me Button
    const btnSurprise = document.getElementById('btn-surprise-me');
    if (btnSurprise) {
      btnSurprise.addEventListener('click', () => {
        this.launchPreset('surprise_me');
      });
    }

    // Play Tool Palette Buttons
    const playTools = ['tool-play-add', 'tool-play-select', 'tool-play-accelerate', 'tool-play-compare', 'tool-play-save'];
    playTools.forEach(id => {
      const btn = document.getElementById(id);
      if (btn) {
        btn.addEventListener('click', () => {
          document.querySelectorAll('.btn-play-tool').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          this.handleToolSelect(btn.dataset.tool);
        });
      }
    });

    // Cancer Battle Live Card Actions
    const btnApplyGem = document.getElementById('btn-play-apply-gemcitabine');
    if (btnApplyGem) {
      btnApplyGem.addEventListener('click', () => {
        this.app.apiPost('/tumor/introduce-compound', {
          compound_query: 'Gemcitabine',
          applied_concentration_um: 5.0,
          injection_region: 'vascular',
        }).then(res => {
          this.app.showToast('Introduced Gemcitabine (5 µM) to PDAC Microenvironment');
        });
      });
    }

    const btnRunControl = document.getElementById('btn-play-run-control-comp');
    if (btnRunControl) {
      btnRunControl.addEventListener('click', () => {
        this.app.apiPost('/tumor/controlled-comparison', {
          compound_query: 'Gemcitabine',
          applied_concentration_um: 5.0,
          duration_hours: 48.0,
        }).then(res => {
          this.app.showToast('Controlled Comparison Complete: Gemcitabine vs Untreated Control');
        });
      });
    }

    // Quick links to LAB views
    const linkToBio = document.getElementById('btn-play-to-lab-bio');
    if (linkToBio) {
      linkToBio.addEventListener('click', () => {
        this.app.switchMode('LAB');
        this.app.switchLabTab('tab-biomarkers');
      });
    }

    const linkToMol = document.getElementById('btn-play-to-lab-mol');
    if (linkToMol) {
      linkToMol.addEventListener('click', () => {
        this.app.switchMode('LAB');
        this.app.switchLabTab('tab-molecular');
      });
    }

    // Explain in Plain English Button
    const btnExplain = document.getElementById('btn-play-explain-compound');
    if (btnExplain) {
      btnExplain.addEventListener('click', () => {
        const name = document.getElementById('play-cmp-name').textContent;
        if (!name || name === '--') {
          this.app.showToast('Select an entity or molecule first!');
          return;
        }
        this.app.apiPost('/assistant/ask', {
          question: `Explain how the molecule ${name} works in plain English without complex jargon.`,
        }).then(res => {
          const answer = res.data?.answer || 'Explanation unavailable.';
          const modal = document.getElementById('modal-explain-compound');
          const heading = document.getElementById('explain-compound-heading');
          const body = document.getElementById('explain-compound-body');
          const tag = document.getElementById('explain-compound-tag');
          if (modal && heading && body) {
            heading.textContent = name;
            body.textContent = answer;
            if (tag && res.data?.epistemic_category) tag.textContent = `Epistemic: ${res.data.epistemic_category}`;
            modal.classList.remove('hidden');
          }
        });
      });
    }

    // Jump Presets
    const jumps = [
      { id: 'btn-play-jump-10', preset: '10_gen', msg: 'Jumped 10 Generations!' },
      { id: 'btn-play-jump-100', preset: '100_gen', msg: 'Jumped 100 Generations!' },
      { id: 'btn-play-jump-1yr', preset: '1_yr', msg: 'Jumped 1 Virtual Year!' },
      { id: 'btn-play-jump-10yr', preset: '10_yr', msg: 'Jumped 10 Virtual Years!' },
    ];
    for (const j of jumps) {
      const btn = document.getElementById(j.id);
      if (btn) {
        btn.addEventListener('click', () => {
          this.app.apiPost('/world/advance-preset', { preset: j.preset }).then(() => {
            this.app.showToast(j.msg);
          });
        });
      }
    }

    // Viewport Target Toggle (Universe vs Tumor)
    const btnUniv = document.getElementById('btn-toggle-universe-view');
    const btnTumor = document.getElementById('btn-toggle-tumor-view');
    const cvUniv = document.getElementById('play-viewport-canvas');
    const cvTumor = document.getElementById('play-tumor-canvas');

    if (btnUniv && btnTumor) {
      btnUniv.addEventListener('click', () => {
        btnUniv.classList.add('active');
        btnTumor.classList.remove('active');
        if (cvUniv) cvUniv.classList.remove('hidden');
        if (cvTumor) cvTumor.classList.add('hidden');
        this.app.playTarget = 'universe';
        this.app.showToast('Play Viewport: Chemical Universe');
      });
      btnTumor.addEventListener('click', () => {
        btnTumor.classList.add('active');
        btnUniv.classList.remove('active');
        if (cvTumor) cvTumor.classList.remove('hidden');
        if (cvUniv) cvUniv.classList.add('hidden');
        this.app.playTarget = 'tumor';
        this.app.showToast('Play Viewport: Cancer Battle');
      });
    }

    // Toast Close Button
    const btnCloseToast = document.getElementById('btn-close-toast');
    if (btnCloseToast) {
      btnCloseToast.addEventListener('click', () => {
        document.getElementById('evolution-summary-toast').classList.add('hidden');
      });
    }

    // Tutorial Navigation Buttons
    const btnTutNext = document.getElementById('btn-tut-next');
    if (btnTutNext) {
      btnTutNext.addEventListener('click', () => this.nextTutorialStep());
    }
    const btnTutPrev = document.getElementById('btn-tut-prev');
    if (btnTutPrev) {
      btnTutPrev.addEventListener('click', () => this.prevTutorialStep());
    }
    const btnTutSkip = document.getElementById('btn-tut-skip');
    if (btnTutSkip) {
      btnTutSkip.addEventListener('click', () => this.closeTutorial());
    }

    // Glossary Search
    const glossaryInput = document.getElementById('glossary-search-input');
    if (glossaryInput) {
      glossaryInput.addEventListener('input', (e) => {
        this.filterGlossary(e.target.value.toLowerCase());
      });
    }
  }

  launchPreset(preset) {
    if (preset === 'evocure_001') {
      this.app.showToast('Starting EVOCURE-001 Benchmark Scenario...');
      this.app.apiPost('/discovery/evocure-001', {}).then(res => {
        this.app.showToast('EVOCURE-001 Reference Benchmark Finished!');
      });
      return;
    }

    this.app.showToast(`Launching ${preset.replace('_', ' ')}...`);
    this.app.apiPost('/world/reset', { preset: preset }).then(res => {
      // Auto-start playback
      this.app.apiPost('/playback', {
        action: 'play',
        speed_multiplier: 1,
        target: preset === 'cancer_battle' ? 'tumor' : 'universe',
      });
      this.app.showToast('Simulation running!');
    });
  }

  handleToolSelect(tool) {
    if (tool === 'accelerate') {
      this.app.apiPost('/world/advance-preset', { preset: '100_gen' }).then(() => {
        this.app.showToast('Evolutionary time accelerated by 100 Generations!');
      });
    } else if (tool === 'compare') {
      this.app.switchMode('LAB');
      this.app.switchLabTab('tab-comparison');
    } else if (tool === 'save') {
      this.app.apiPost('/checkpoints/create', { label: 'Quick Save' }).then(() => {
        this.app.showToast('Bookmark checkpoint saved!');
      });
    }
  }

  updatePlayMetrics(state) {
    if (!state) return;

    // Update 4-Metric Cancer Battle live status
    const tumor = state.tumor;
    if (tumor) {
      const summary = tumor.summary_metrics || {};
      const zones = tumor.zones || {};
      const elTumor = document.getElementById('play-val-tumor-cells');
      const elRes = document.getElementById('play-val-resistant-pct');
      const elSpared = document.getElementById('play-val-healthy-spared');
      const elHypoxia = document.getElementById('play-val-hypoxia-pct');

      if (elTumor) elTumor.textContent = summary.viable_tumor_cells ?? summary.viable_cells ?? '--';
      if (elRes) elRes.textContent = `${Math.round((summary.resistant_fraction || 0) * 100)}%`;
      if (elSpared) {
        const counts = tumor.cell_counts_by_type || {};
        const healthy = counts.EPITHELIAL_HEALTHY ?? 35;
        elSpared.textContent = `${healthy} cells`;
      }
      if (elHypoxia) elHypoxia.textContent = `${Math.round(zones.hypoxic_area_pct || 19.5)}%`;
    }

    // Update Replicators badge
    const badge = document.getElementById('play-entity-badge');
    if (badge) {
      const total = state.entity_total_count ?? (state.entities?.length || 0);
      badge.textContent = `Replicators: ${total}`;
    }

    // Trigger Periodic Milestone Toast every 10 generations or when new lineages emerge
    const gen = state.clocks?.evolutionary_generation || 0;
    if (gen > 0 && gen % 10 === 0 && this.lastAnnouncedGen !== gen) {
      this.lastAnnouncedGen = gen;
      this.showMilestoneToast(
        `Generation ${gen} Milestone Reached!`,
        `The ecosystem has sustained replicators across ${state.clocks?.virtual_years || 0} virtual years.`
      );
    }
  }

  showMilestoneToast(title, body) {
    const toast = document.getElementById('evolution-summary-toast');
    const elTitle = document.getElementById('toast-title');
    const elBody = document.getElementById('toast-body');
    if (toast && elTitle && elBody) {
      elTitle.textContent = title;
      elBody.textContent = body;
      toast.classList.remove('hidden');
      setTimeout(() => toast.classList.add('hidden'), 8000);
    }
  }

  displaySelectedCompound(compound) {
    if (!compound) return;
    const elName = document.getElementById('play-cmp-name');
    const elFormula = document.getElementById('play-cmp-formula');
    const elRole = document.getElementById('play-cmp-role');
    const elMW = document.getElementById('play-cmp-mw');
    const svgBox = document.getElementById('play-compound-svg-container');

    if (elName) elName.textContent = compound.name || compound.common_name || 'Selected Entity';
    if (elFormula) elFormula.textContent = compound.formula || '--';
    if (elRole) elRole.textContent = compound.ecological_role || compound.role || 'autotroph';
    if (elMW) elMW.textContent = compound.mol_weight ? compound.mol_weight.toFixed(1) : '--';

    if (svgBox && compound.svg_2d) {
      svgBox.innerHTML = compound.svg_2d;
    }
  }

  // Tutorial Stepper Logic
  loadTutorialSteps(steps) {
    this.tutorialStepsData = steps || [];
    this.totalTutorialSteps = this.tutorialStepsData.length || 10;
  }

  showTutorial() {
    this.currentTutorialStep = 1;
    this._renderTutorialStep();
    const modal = document.getElementById('modal-tutorial');
    if (modal) modal.classList.remove('hidden');
  }

  nextTutorialStep() {
    if (this.currentTutorialStep < this.totalTutorialSteps) {
      this.currentTutorialStep++;
      this._renderTutorialStep();
    } else {
      this.closeTutorial();
    }
  }

  prevTutorialStep() {
    if (this.currentTutorialStep > 1) {
      this.currentTutorialStep--;
      this._renderTutorialStep();
    }
  }

  closeTutorial() {
    const modal = document.getElementById('modal-tutorial');
    if (modal) modal.classList.add('hidden');
  }

  _renderTutorialStep() {
    const step = this.tutorialStepsData[this.currentTutorialStep - 1];
    if (!step) return;

    const elTitle = document.getElementById('tutorial-step-title');
    const elBody = document.getElementById('tutorial-step-body');
    const elCounter = document.getElementById('tutorial-step-counter');
    const elFill = document.getElementById('tutorial-progress-fill');
    const btnPrev = document.getElementById('btn-tut-prev');
    const btnNext = document.getElementById('btn-tut-next');

    if (elTitle) elTitle.textContent = step.title;
    if (elBody) elBody.textContent = step.body;
    if (elCounter) elCounter.textContent = `Step ${this.currentTutorialStep} of ${this.totalTutorialSteps}`;
    if (elFill) elFill.style.width = `${(this.currentTutorialStep / this.totalTutorialSteps) * 100}%`;

    if (btnPrev) btnPrev.disabled = (this.currentTutorialStep === 1);
    if (btnNext) {
      btnNext.textContent = (this.currentTutorialStep === this.totalTutorialSteps) ? 'Finish Tutorial' : 'Next Step →';
    }
  }

  // Missions & Glossary Population
  loadMissions(missions, achievements) {
    this.missionsData = missions || [];
    const container = document.getElementById('missions-container');
    if (!container) return;
    container.innerHTML = '';

    for (const m of this.missionsData) {
      const card = document.createElement('div');
      card.className = 'mission-card';
      card.innerHTML = `
        <div class="mission-title-row">
          <strong>${m.title}</strong>
          <span class="mission-badge">${m.preset_mode}</span>
        </div>
        <p style="font-size: 0.78rem; color: var(--text-secondary); margin: 4px 0 8px 0;">${m.subtitle}</p>
        <div style="font-size: 0.72rem; color: var(--text-muted);">
          <div><strong>Captures:</strong> ${m.what_model_captures}</div>
          <div style="margin-top: 3px;"><strong>Limitations:</strong> ${m.what_remains_unknown}</div>
        </div>
      `;
      container.appendChild(card);
    }

    const achContainer = document.getElementById('achievements-container');
    if (achContainer && achievements) {
      achContainer.innerHTML = '';
      for (const a of achievements) {
        const item = document.createElement('div');
        item.className = 'mission-card';
        item.innerHTML = `
          <div class="mission-title-row">
            <strong>${a.title}</strong>
            <span class="mission-badge">${a.category}</span>
          </div>
          <p style="font-size: 0.78rem; color: var(--text-secondary);">${a.description}</p>
        `;
        achContainer.appendChild(item);
      }
    }
  }

  loadGlossary(glossary) {
    this.glossaryData = glossary || [];
    this.filterGlossary('');
  }

  filterGlossary(query) {
    const container = document.getElementById('glossary-container');
    if (!container) return;
    container.innerHTML = '';

    const filtered = this.glossaryData.filter(item => 
      !query || item.term.toLowerCase().includes(query) || item.simple.toLowerCase().includes(query)
    );

    for (const g of filtered) {
      const card = document.createElement('div');
      card.className = 'glossary-term-card';
      card.innerHTML = `
        <div class="term-title">${g.term}</div>
        <div class="term-simple">${g.simple}</div>
        <div class="term-technical"><strong>Technical Definition:</strong> ${g.technical}</div>
      `;
      container.appendChild(card);
    }
  }
}

window.PlayModeManager = PlayModeManager;
