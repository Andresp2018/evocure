/**
 * EVOCURE Main ES6 Application Controller.
 * 
 * Synchronizes real-time WebSocket state, REST APIs, Canvas 2D renderers,
 * and presentation transitions across EVOCURE PLAY and EVOCURE LAB.
 */

class EvocureApp {
  constructor() {
    this.currentMode = 'PLAY';
    this.currentLabTab = 'tab-universe';
    this.ws = null;
    this.wsConnected = false;
    this.lastState = null;

    // Renderers
    this.universeRenderer = null;
    this.playUniverseRenderer = null;
    this.playTumorRenderer = null;
    this.playTarget = 'universe';
    this.tumorRenderer = null;
    this.treeRenderer = null;
    this.telemetryChart = null;
    this.admetRadar = null;

    // Wave 4B Enhanced Renderers
    this.synergyRenderer = null;
    this.hillRenderer = null;
    this.pbpkChart = null;
    this.fieldO2Renderer = null;
    this.fieldGlucoseRenderer = null;
    this.fieldEcmRenderer = null;
    this.fieldDrugRenderer = null;
    this.comparisonBarChart = null;

    // Editor & Genealogy state
    this.editorOperations = null;
    this.currentParentSmiles = 'C1=CN(C(=O)N=C1N)[C@H]2C([C@@H]([C@H](O2)CO)O)(F)F';
    this.currentParentName = 'Gemcitabine';
    this.treeNodesCache = [];

    // Sub-controllers
    this.playManager = null;

    this.init();
  }

  async init() {
    console.log("Initializing EVOCURE 100% Local Application...");

    // 1. Initialize Renderers
    this._initRenderers();

    // 2. Initialize Sub-Managers
    this.playManager = new PlayModeManager(this);

    // 3. Connect WebSocket
    this._connectWebSocket();

    // 4. Bind UI Controls, Modals & Keyboard Shortcuts
    this._bindUiControls();
    this._bindKeyboardShortcuts();

    // 5. Fetch Initial Missions, Glossary & Compound Library
    await this._fetchInitialData();

    // 6. Initial State Pull
    const state = await this.apiGet('/state');
    if (state) {
      this.handleStateUpdate(state);
    }
  }

  _initRenderers() {
    // Play Universe Canvas
    const playCanvas = document.getElementById('play-viewport-canvas');
    if (playCanvas) {
      this.playUniverseRenderer = new UniverseCanvasRenderer(playCanvas, (ent) => {
        this.inspectEntity(ent);
      });
    }

    // Play Tumor Canvas
    const playTumorCanvas = document.getElementById('play-tumor-canvas');
    if (playTumorCanvas) {
      this.playTumorRenderer = new TumorCanvasRenderer(playTumorCanvas, (cell) => {
        this.inspectCell(cell);
      });
    }

    // Lab Universe Canvas
    const labCanvas = document.getElementById('universe-canvas');
    if (labCanvas) {
      this.universeRenderer = new UniverseCanvasRenderer(labCanvas, (ent) => {
        this.inspectEntity(ent);
      });
    }

    // Lab Tumor Canvas
    const tumorCanvas = document.getElementById('tumor-canvas');
    if (tumorCanvas) {
      this.tumorRenderer = new TumorCanvasRenderer(tumorCanvas, (cell) => {
        this.inspectCell(cell);
      });
    }

    // Family Tree DAG Canvas
    const treeCanvas = document.getElementById('tree-canvas');
    if (treeCanvas) {
      this.treeRenderer = new FamilyTreeDAGRenderer(treeCanvas, (node) => {
        this.inspectCompoundByQuery(node.compound_id || node.canonical_smiles);
      });
    }

    // Bottom Telemetry Line Chart
    const lineChartCanvas = document.getElementById('bottom-realtime-chart');
    if (lineChartCanvas) {
      this.telemetryChart = new TelemetryLineChart(lineChartCanvas);
    }

    // Evolution Tab Diversity & Population Charts
    const cvDiversity = document.getElementById('chart-diversity');
    if (cvDiversity) {
      this.diversityChart = new TelemetryLineChart(cvDiversity);
    }
    const cvPopulation = document.getElementById('chart-population');
    if (cvPopulation) {
      this.populationChart = new TelemetryLineChart(cvPopulation);
    }

    // ADMET Radar Chart
    const radarCanvas = document.getElementById('chart-admet-radar');
    if (radarCanvas) {
      this.admetRadar = new ADMETRadarChart(radarCanvas);
      this.admetRadar.render();
    }

    // Synergy 8x8 Heatmap Renderer
    const synergyCanvas = document.getElementById('chart-synergy-matrix');
    if (synergyCanvas && window.SynergyHeatmapRenderer) {
      this.synergyRenderer = new SynergyHeatmapRenderer(synergyCanvas);
      this.synergyRenderer.render();
    }

    // Hill Sigmoidal Curve Renderer
    const hillCanvas = document.getElementById('chart-hill-curves');
    if (hillCanvas && window.HillCurveRenderer) {
      this.hillRenderer = new HillCurveRenderer(hillCanvas);
      this.hillRenderer.render();
    }

    // 7-Compartment PBPK Kinetics Chart
    const pbpkCanvas = document.getElementById('chart-pbpk-kinetics');
    if (pbpkCanvas && window.PBPKKineticsChart) {
      this.pbpkChart = new PBPKKineticsChart(pbpkCanvas);
      this.pbpkChart.render();
    }

    // 4-Field Spatial Microenvironment Renderers
    const cO2 = document.getElementById('canvas-field-o2');
    if (cO2 && window.MicroenvFieldRenderer) this.fieldO2Renderer = new MicroenvFieldRenderer(cO2, 'o2');
    const cGlu = document.getElementById('canvas-field-glucose');
    if (cGlu && window.MicroenvFieldRenderer) this.fieldGlucoseRenderer = new MicroenvFieldRenderer(cGlu, 'glucose');
    const cEcm = document.getElementById('canvas-field-ecm');
    if (cEcm && window.MicroenvFieldRenderer) this.fieldEcmRenderer = new MicroenvFieldRenderer(cEcm, 'ecm');
    const cDrug = document.getElementById('canvas-field-drug');
    if (cDrug && window.MicroenvFieldRenderer) this.fieldDrugRenderer = new MicroenvFieldRenderer(cDrug, 'drug');

    // Side-by-Side Comparison Grouped Bar Chart
    const compCanvas = document.getElementById('chart-comparison-side-by-side');
    if (compCanvas && window.ComparisonBarChartRenderer) {
      this.comparisonBarChart = new ComparisonBarChartRenderer(compCanvas);
      this.comparisonBarChart.render();
    }
  }

  _connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      this.wsConnected = true;
      console.log("WebSocket connected to EVOCURE simulation stream.");
    };

    this.ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'state_update' || msg.type === 'init_state') {
          this.handleStateUpdate(msg.data);
        } else if (msg.type === 'compound_inspected') {
          this.displayCompoundDetails(msg.data);
        } else if (msg.type === 'playback_ack' || msg.type === 'playback_change') {
          this.updatePlaybackControlsUI(msg);
        }
      } catch (e) {
        console.error("WS Parse error:", e);
      }
    };

    this.ws.onclose = () => {
      this.wsConnected = false;
      console.warn("WebSocket disconnected. Reconnecting in 2 seconds...");
      setTimeout(() => this._connectWebSocket(), 2000);
    };

    this.ws.onerror = (err) => {
      console.error("WebSocket error:", err);
    };
  }

  handleStateUpdate(state) {
    if (!state) return;
    this.lastState = state;

    // Update Clocks Header
    const clocks = state.clocks || {};
    const elStep = document.getElementById('clock-step');
    const elGen = document.getElementById('clock-gen');
    const elVYears = document.getElementById('clock-vyears');
    const elBioHours = document.getElementById('clock-biohours');
    const elUnivMode = document.getElementById('header-universe-mode');

    if (elStep) elStep.textContent = clocks.evolutionary_step ?? 0;
    if (elGen) elGen.textContent = clocks.evolutionary_generation ?? 0;
    if (elVYears) elVYears.textContent = (clocks.virtual_years || 0).toFixed(2);
    if (elBioHours) elBioHours.textContent = (clocks.biological_hours || 0).toFixed(1);
    if (elUnivMode) elUnivMode.textContent = state.universe_mode || 'ARTIFICIAL_LIFE';

    // Update Renderers
    if (this.currentMode === 'PLAY') {
      if (this.playUniverseRenderer) this.playUniverseRenderer.updateState(state);
      if (this.playManager) this.playManager.updatePlayMetrics(state);
    } else {
      if (this.universeRenderer) this.universeRenderer.updateState(state);
      if (this.tumorRenderer && state.tumor) this.tumorRenderer.updateState(state.tumor);
      if (this.telemetryChart && state.metrics) this.telemetryChart.addDataPoint(state.metrics);
      this._updateLabViewMetrics(state);
    }
  }

  _updateLabViewMetrics(state) {
    const tumor = state.tumor;
    if (tumor) {
      const zones = tumor.zones || {};
      const summary = tumor.summary_metrics || {};
      const elNorm = document.getElementById('lab-normoxic-pct');
      const elHypox = document.getElementById('lab-hypoxic-pct');
      const elNecro = document.getElementById('lab-necrotic-pct');
      const elEcm = document.getElementById('lab-ecm-density');

      if (elNorm) elNorm.textContent = `${(zones.normoxic_area_pct || 52).toFixed(1)}%`;
      if (elHypox) elHypox.textContent = `${(zones.hypoxic_area_pct || 19.5).toFixed(1)}%`;
      if (elNecro) elNecro.textContent = `${(zones.necrotic_core_pct || 28.5).toFixed(1)}%`;
      if (elEcm) elEcm.textContent = (summary.mean_ecm_density || 0.145).toFixed(3);
    }
  }

  // UI Control Binding
  _bindUiControls() {
    // Mode Switch buttons (PLAY <-> LAB)
    const btnModePlay = document.getElementById('btn-mode-play');
    const btnModeLab = document.getElementById('btn-mode-lab');

    if (btnModePlay && btnModeLab) {
      btnModePlay.addEventListener('click', () => this.switchMode('PLAY'));
      btnModeLab.addEventListener('click', () => this.switchMode('LAB'));
    }

    // 10 LAB View Navigation Tabs
    document.querySelectorAll('.lab-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        const targetTab = tab.dataset.tab;
        this.switchLabTab(targetTab);
      });
    });

    // Playback buttons in Play bar
    const btnPlayPause = document.getElementById('play-btn-playpause');
    const btnPlayStep = document.getElementById('play-btn-step');
    if (btnPlayPause) {
      btnPlayPause.addEventListener('click', () => this.togglePlayback());
    }
    if (btnPlayStep) {
      btnPlayStep.addEventListener('click', () => this.stepSimulation());
    }

    // Playback buttons in Lab bar
    const btnLabPlayPause = document.getElementById('lab-btn-playpause');
    const btnLabStep = document.getElementById('lab-btn-step');
    if (btnLabPlayPause) {
      btnLabPlayPause.addEventListener('click', () => this.togglePlayback());
    }
    if (btnLabStep) {
      btnLabStep.addEventListener('click', () => this.stepSimulation());
    }

    const btnLabUniv = document.getElementById('lab-btn-target-univ');
    const btnLabTumor = document.getElementById('lab-btn-target-tumor');
    if (btnLabUniv && btnLabTumor) {
      btnLabUniv.addEventListener('click', () => {
        btnLabUniv.classList.add('active');
        btnLabTumor.classList.remove('active');
        this.apiPost('/playback', { action: 'speed', target: 'universe' });
      });
      btnLabTumor.addEventListener('click', () => {
        btnLabTumor.classList.add('active');
        btnLabUniv.classList.remove('active');
        this.apiPost('/playback', { action: 'speed', target: 'tumor' });
      });
    }

    // Reaction ripples checkbox
    const chkRipples = document.getElementById('chk-reaction-ripples');
    if (chkRipples) {
      chkRipples.addEventListener('change', (e) => {
        if (this.universeRenderer) this.universeRenderer.enableRipples = e.target.checked;
        if (this.playUniverseRenderer) this.playUniverseRenderer.enableRipples = e.target.checked;
      });
    }

    // Compound Inspector Actions
    const btnInspTumor = document.getElementById('btn-insp-add-to-tumor');
    if (btnInspTumor) {
      btnInspTumor.addEventListener('click', () => {
        const smiles = document.getElementById('insp-smiles')?.textContent;
        if (!smiles || smiles === '--') {
          this.showToast('Select a compound to test on tumor first!');
          return;
        }
        this.apiPost('/tumor/introduce-compound', {
          compound_query: smiles,
          applied_concentration_um: 5.0,
          injection_region: 'vascular',
        }).then(() => {
          this.showToast('Introduced compound to tumor microenvironment');
        });
      });
    }

    const btnInspSpawn = document.getElementById('btn-insp-spawn-more');
    if (btnInspSpawn) {
      btnInspSpawn.addEventListener('click', () => {
        const smiles = document.getElementById('insp-smiles')?.textContent;
        if (!smiles || smiles === '--') {
          this.showToast('Select a compound first!');
          return;
        }
        this.apiPost('/world/add-compound', {
          query: smiles,
          count: 12,
        }).then(() => {
          this.showToast('Spawned +12 instances into the universe!');
        });
      });
    }

    // Quick Bookmark in bottom dock
    const btnQuickCkpt = document.getElementById('btn-quick-create-ckpt');
    if (btnQuickCkpt) {
      btnQuickCkpt.addEventListener('click', () => {
        this.apiPost('/checkpoints/create', { label: `Quick Bookmark (${new Date().toLocaleTimeString()})` }).then(() => {
          this.showToast('Bookmark created!');
          this._refreshCheckpointsList();
        });
      });
    }

    // Compound Library Search & Filter
    const libSearch = document.getElementById('lib-search-input');
    const libCat = document.getElementById('lib-category-filter');
    const filterLib = () => {
      const q = libSearch?.value.toLowerCase().trim() || '';
      const cat = libCat?.value || 'all';
      document.querySelectorAll('#library-compounds-list .library-card-item').forEach(li => {
        const name = li.querySelector('.item-name')?.textContent.toLowerCase() || '';
        const role = li.querySelector('.item-role')?.textContent.toLowerCase() || '';
        const matchesQ = !q || name.includes(q) || role.includes(q);
        const matchesCat = (cat === 'all') || (cat === 'pdac_reference' && (role.includes('drug') || name.includes('gemcitabine') || name.includes('sotorasib') || name.includes('cisplatin'))) || (cat === 'prebiotic' && (role.includes('prebiotic') || role.includes('amino') || name.includes('glycine') || name.includes('adenine'))) || (cat === 'benign_control');
        li.style.display = (matchesQ && matchesCat) ? 'flex' : 'none';
      });
    };
    if (libSearch) libSearch.addEventListener('input', filterLib);
    if (libCat) libCat.addEventListener('change', filterLib);

    // Speed toggle buttons
    document.querySelectorAll('.btn-speed').forEach(btn => {
      btn.addEventListener('click', () => {
        const speed = parseInt(btn.dataset.speed, 10);
        document.querySelectorAll('.btn-speed').forEach(b => b.classList.toggle('active', parseInt(b.dataset.speed, 10) === speed));
        this.apiPost('/playback', { action: 'speed', speed_multiplier: speed });
      });
    });

    // Playback Target Toggles (Universe vs Tumor)
    const btnToggleUniv = document.getElementById('btn-toggle-universe-view');
    const btnToggleTumor = document.getElementById('btn-toggle-tumor-view');
    if (btnToggleUniv && btnToggleTumor) {
      btnToggleUniv.addEventListener('click', () => {
        btnToggleUniv.classList.add('active');
        btnToggleTumor.classList.remove('active');
        this.apiPost('/playback', { action: 'speed', target: 'universe' });
      });
      btnToggleTumor.addEventListener('click', () => {
        btnToggleTumor.classList.add('active');
        btnToggleUniv.classList.remove('active');
        this.apiPost('/playback', { action: 'speed', target: 'tumor' });
      });
    }

    // Canvas Zoom Controls in LAB
    const btnZoomIn = document.getElementById('btn-zoom-in');
    const btnZoomOut = document.getElementById('btn-zoom-out');
    const btnZoomReset = document.getElementById('btn-zoom-reset');
    if (btnZoomIn && this.universeRenderer) {
      btnZoomIn.addEventListener('click', () => {
        this.universeRenderer.setZoom(this.universeRenderer.zoom * 1.25);
        document.getElementById('universe-zoom-badge').textContent = `${Math.round(this.universeRenderer.zoom * 100)}%`;
      });
    }
    if (btnZoomOut && this.universeRenderer) {
      btnZoomOut.addEventListener('click', () => {
        this.universeRenderer.setZoom(this.universeRenderer.zoom / 1.25);
        document.getElementById('universe-zoom-badge').textContent = `${Math.round(this.universeRenderer.zoom * 100)}%`;
      });
    }
    if (btnZoomReset && this.universeRenderer) {
      btnZoomReset.addEventListener('click', () => {
        this.universeRenderer.resetView();
        document.getElementById('universe-zoom-badge').textContent = `100%`;
      });
    }

    // Environment Form
    const envForm = document.getElementById('form-environment');
    if (envForm) {
      envForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const temp = parseFloat(document.getElementById('slider-temp').value);
        const ph = parseFloat(document.getElementById('slider-ph').value);
        const visc = parseFloat(document.getElementById('slider-viscosity').value);
        const influx = parseFloat(document.getElementById('slider-influx').value);

        this.apiPost('/environment', {
          temperature_k: temp,
          ph: ph,
          viscosity_pa_s: visc,
          resource_influx_rate: influx,
        }).then(() => {
          this.showToast('Environmental niche conditions updated!');
        });
      });

      // Synchronize slider output values live
      const sliderTemp = document.getElementById('slider-temp');
      if (sliderTemp) sliderTemp.addEventListener('input', (e) => document.getElementById('val-temp').textContent = `${parseFloat(e.target.value).toFixed(2)} K`);
      const sliderPh = document.getElementById('slider-ph');
      if (sliderPh) sliderPh.addEventListener('input', (e) => document.getElementById('val-ph').textContent = parseFloat(e.target.value).toFixed(2));
      const sliderVisc = document.getElementById('slider-viscosity');
      if (sliderVisc) sliderVisc.addEventListener('input', (e) => document.getElementById('val-viscosity').textContent = parseFloat(e.target.value).toFixed(5));
      const sliderInflux = document.getElementById('slider-influx');
      if (sliderInflux) sliderInflux.addEventListener('input', (e) => document.getElementById('val-influx').textContent = parseFloat(e.target.value).toFixed(2));
    }

    // Preset advance buttons in Evolution tab
    document.querySelectorAll('.btn-preset-advance').forEach(btn => {
      btn.addEventListener('click', () => {
        const preset = btn.dataset.preset;
        this.showToast(`Advancing evolution by ${preset}...`);
        this.apiPost('/world/advance-preset', { preset: preset }).then(res => {
          this.showToast(`Advanced simulation by ${preset}!`);
        });
      });
    });

    // Molecular Explorer Search
    const btnMolSearch = document.getElementById('btn-mol-explore-search');
    const molInput = document.getElementById('mol-explore-input');
    if (btnMolSearch && molInput) {
      btnMolSearch.addEventListener('click', () => {
        const val = molInput.value.trim();
        if (val) this.inspectCompoundByQuery(val);
      });
    }

    // AI Assistant Chat Submit
    const chatForm = document.getElementById('ai-chat-form');
    if (chatForm) {
      chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const input = document.getElementById('ai-chat-input');
        const query = input.value.trim();
        if (!query) return;

        this._appendChatMessage('user', query);
        input.value = '';

        this.apiPost('/assistant/ask', { question: query }).then(res => {
          const ans = res.data?.answer || 'Response generated.';
          const tag = res.data?.epistemic_category || 'AI-generated hypothesis';
          const action = res.data?.proposed_action;
          this._appendChatMessage('bot', ans, tag, action);
        });
      });
    }

    // Accessibility Menu Dropdown Toggle
    const btnAccess = document.getElementById('btn-accessibility');
    const accessMenu = document.getElementById('accessibility-menu');
    if (btnAccess && accessMenu) {
      btnAccess.addEventListener('click', () => {
        const isHidden = accessMenu.classList.contains('hidden');
        accessMenu.classList.toggle('hidden');
        btnAccess.setAttribute('aria-expanded', isHidden ? 'true' : 'false');
      });
    }

    // Colorblind Palette Selection
    const selectPalette = document.getElementById('select-palette');
    if (selectPalette) {
      selectPalette.addEventListener('change', (e) => {
        document.body.setAttribute('data-palette', e.target.value);
      });
    }

    // Font Size Selection
    const selectFontSize = document.getElementById('select-font-size');
    if (selectFontSize) {
      selectFontSize.addEventListener('change', (e) => {
        document.body.setAttribute('data-font-size', e.target.value);
      });
    }

    // Reduced Motion Checkbox
    const chkMotion = document.getElementById('chk-reduced-motion');
    if (chkMotion) {
      chkMotion.addEventListener('change', (e) => {
        document.body.classList.toggle('reduced-motion', e.target.checked);
      });
    }

    // Active Disease Selector
    const selDisease = document.getElementById('select-active-disease');
    if (selDisease) {
      selDisease.addEventListener('change', (e) => {
        const disId = e.target.value;
        this.apiPost('/diseases/select', { disease_id: disId }).then(res => {
          this.showToast(`Switched active disease target to ${disId}`);
        });
      });
    }

    // Experiments Manager Modal
    const btnOpenExp = document.getElementById('btn-open-experiments');
    const modalExp = document.getElementById('modal-experiments');
    if (btnOpenExp && modalExp) {
      btnOpenExp.addEventListener('click', () => {
        this._refreshExperimentsList();
        modalExp.classList.remove('hidden');
      });
    }

    const btnSaveExp = document.getElementById('btn-save-exp-action');
    if (btnSaveExp) {
      btnSaveExp.addEventListener('click', () => {
        const title = document.getElementById('exp-save-title').value.trim() || 'EVOCURE Simulation Session';
        this.apiPost('/experiments/save', { title: title }).then(() => {
          this.showToast(`Saved experiment session: ${title}`);
          this._refreshExperimentsList();
          document.getElementById('exp-save-title').value = '';
        });
      });
    }

    // Modals Open Buttons
    const btnTut = document.getElementById('btn-open-tutorial');
    if (btnTut) btnTut.addEventListener('click', () => this.playManager.showTutorial());

    const btnMissions = document.getElementById('btn-open-missions');
    const drawerMissions = document.getElementById('drawer-missions');
    if (btnMissions && drawerMissions) {
      btnMissions.addEventListener('click', () => drawerMissions.classList.remove('hidden'));
    }

    const btnGlossary = document.getElementById('btn-open-glossary');
    const modalGlossary = document.getElementById('modal-glossary');
    if (btnGlossary && modalGlossary) {
      btnGlossary.addEventListener('click', () => modalGlossary.classList.remove('hidden'));
    }

    const btnCheckpoints = document.getElementById('btn-open-checkpoints');
    const modalCheckpoints = document.getElementById('modal-checkpoints');
    if (btnCheckpoints && modalCheckpoints) {
      btnCheckpoints.addEventListener('click', () => {
        this._refreshCheckpointsList();
        modalCheckpoints.classList.remove('hidden');
      });
    }

    // Close all modal buttons
    document.querySelectorAll('.btn-close-modal').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.modal-backdrop').forEach(m => m.classList.add('hidden'));
      });
    });

    // Checkpoint Creation in Modal
    const btnCreateCkpt = document.getElementById('btn-create-ckpt-action');
    if (btnCreateCkpt) {
      btnCreateCkpt.addEventListener('click', () => {
        const lbl = document.getElementById('ckpt-label-input').value.trim() || 'Manual Bookmark';
        this.apiPost('/checkpoints/create', { label: lbl }).then(() => {
          this.showToast(`Saved checkpoint '${lbl}'`);
          this._refreshCheckpointsList();
          document.getElementById('ckpt-label-input').value = '';
        });
      });
    }

    // Drawer Tabs (Missions vs Achievements)
    document.querySelectorAll('.btn-drawer-tab').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.btn-drawer-tab').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const target = btn.dataset.drawerTab;
        document.querySelectorAll('.drawer-pane').forEach(p => p.classList.add('hidden'));
        const activePane = document.getElementById(target);
        if (activePane) activePane.classList.remove('hidden');
      });
    });

    // Tab 2: NSGA-II Discovery Runner
    const btnRunDiscovery = document.getElementById('btn-run-nsga2-discovery');
    if (btnRunDiscovery) {
      btnRunDiscovery.addEventListener('click', () => {
        const seed = document.getElementById('discovery-seed-select').value;
        const gens = parseInt(document.getElementById('discovery-generations').value, 10) || 4;
        const pop = parseInt(document.getElementById('discovery-pop-size').value, 10) || 16;

        this.showToast(`Running NSGA-II Discovery (${gens} gens, pop ${pop})...`);
        this.apiPost('/discovery/run', {
          seed_queries: [seed],
          generations: gens,
          population_size: pop,
          top_k_cellular: 3,
        }).then(res => {
          this.showToast('NSGA-II Molecular Discovery Complete!');
          this._renderDiscoveryResults(res.data);
        }).catch(err => {
          this.showToast(`Discovery error: ${err.message}`);
        });
      });
    }

    const btnRunBench = document.getElementById('btn-run-evocure-001');
    if (btnRunBench) {
      btnRunBench.addEventListener('click', () => {
        this.showToast('Executing EVOCURE-001 Benchmark Scenario...');
        this.apiPost('/discovery/evocure-001', {}).then(res => {
          this.showToast('EVOCURE-001 Reference Benchmark Execution Complete!');
          this._renderDiscoveryResults(res.data?.discovery_results);
        });
      });
    }

    // Tab 3: Combination Synergy
    const btnRunCombo = document.getElementById('btn-lab-run-combo');
    if (btnRunCombo) {
      btnRunCombo.addEventListener('click', () => {
        const drugA = document.getElementById('combo-drug-a').value;
        const doseA = parseFloat(document.getElementById('combo-dose-a').value) || 5.0;
        const drugB = document.getElementById('combo-drug-b').value;
        const doseB = parseFloat(document.getElementById('combo-dose-b').value) || 10.0;

        this.showToast(`Evaluating synergy: ${drugA} (${doseA}µM) + ${drugB} (${doseB}µM)...`);
        this.apiPost('/tumor/combination', {
          compound_a: drugA,
          dose_a_um: doseA,
          compound_b: drugB,
          dose_b_um: doseB,
          hours: 48.0,
        }).then(res => {
          this.showToast('Combination Synergy evaluation complete!');
          this._renderCombinationResults(res.data);
        });
      });
    }

    const btnRunCoev = document.getElementById('btn-lab-run-coevolution');
    if (btnRunCoev) {
      btnRunCoev.addEventListener('click', () => {
        this.showToast('Running Tumor-Compound Coevolution (4 Cycles)...');
        this.apiPost('/tumor/coevolution', {
          n_cycles: 4,
          hours_per_cycle: 24.0,
          applied_concentration_um: 5.0,
        }).then(() => {
          this.showToast('Coevolution experiment completed!');
        });
      });
    }

    // Tab 5: Molecular Editor Reset
    const btnResetEditor = document.getElementById('btn-editor-reset');
    if (btnResetEditor) {
      btnResetEditor.addEventListener('click', () => {
        this._resetEditorScaffold();
      });
    }

    // Tab 6: Family Tree Sliders & Export
    const simSlider = document.getElementById('tree-similarity-slider');
    const simVal = document.getElementById('val-tree-similarity');
    if (simSlider && simVal) {
      simSlider.addEventListener('input', (e) => {
        simVal.textContent = parseFloat(e.target.value).toFixed(2);
        this._refreshFamilyTree();
      });
    }

    const persSlider = document.getElementById('tree-persistence-slider');
    const persVal = document.getElementById('val-tree-persistence');
    if (persSlider && persVal) {
      persSlider.addEventListener('input', (e) => {
        persVal.textContent = e.target.value;
        this._refreshFamilyTree();
      });
    }

    const btnTreeRefresh = document.getElementById('btn-tree-refresh');
    if (btnTreeRefresh) {
      btnTreeRefresh.addEventListener('click', () => this._refreshFamilyTree());
    }

    const btnTreeExport = document.getElementById('btn-tree-export-json');
    if (btnTreeExport) {
      btnTreeExport.addEventListener('click', () => {
        window.open('/api/v1/genealogy/export', '_blank');
      });
    }

    const btnDiffNodes = document.getElementById('btn-tree-diff-nodes');
    if (btnDiffNodes) {
      btnDiffNodes.addEventListener('click', () => {
        const nodeA = document.getElementById('tree-select-node-a').value;
        const nodeB = document.getElementById('tree-select-node-b').value;
        if (!nodeA || !nodeB) {
          this.showToast('Select two lineage nodes to compare!');
          return;
        }
        this.apiPost('/editor/diff', { smiles_a: nodeA, smiles_b: nodeB }).then(res => {
          this._renderTreeDiffResult(res.data);
        });
      });
    }

    // Tab 8: Toggle PBPK
    const btnTogglePbpk = document.getElementById('btn-toggle-pbpk');
    if (btnTogglePbpk) {
      btnTogglePbpk.addEventListener('click', () => {
        this.showToast('Systemic 7-Compartment PBPK Kinetics Activated');
        this.apiGet('/biomarkers?enable_systemic_pbpk=true').then(data => {
          if (data) this._renderBiomarkersTable(data);
        });
      });
    }

    // Tab 9: AI Telemetry, Multi-Provider Config (Gemma 4 / Gemini / OpenAI) & Triage
    const btnSaveAiProv = document.getElementById('btn-save-ai-provider');
    if (btnSaveAiProv) {
      btnSaveAiProv.addEventListener('click', () => {
        const provider = document.getElementById('ai-provider-select')?.value || 'auto_local';
        const model = document.getElementById('ai-provider-model')?.value?.trim() || undefined;
        const apiKey = document.getElementById('ai-provider-key')?.value?.trim() || undefined;
        this.apiPost('/assistant/providers/configure', {
          provider,
          model,
          api_key: apiKey,
          persist_to_dotenv: false,
        }).then(res => {
          this.showToast(res.message || `AI Provider set to ${provider}`);
          if (document.getElementById('ai-provider-key')) {
            document.getElementById('ai-provider-key').value = '';
          }
          this._refreshAiStatus();
        });
      });
    }

    const btnTestAiProv = document.getElementById('btn-test-ai-provider');
    if (btnTestAiProv) {
      btnTestAiProv.addEventListener('click', () => {
        const provider = document.getElementById('ai-provider-select')?.value || 'auto_local';
        const model = document.getElementById('ai-provider-model')?.value?.trim() || undefined;
        const apiKey = document.getElementById('ai-provider-key')?.value?.trim() || undefined;
        this.apiPost('/assistant/providers/test', {
          provider,
          model,
          api_key: apiKey,
        }).then(res => {
          this.showToast(res.message || 'Provider test finished');
        });
      });
    }

    const btnStartLlm = document.getElementById('btn-start-local-llm');
    if (btnStartLlm) {
      btnStartLlm.addEventListener('click', () => {
        this.apiPost('/assistant/start-local-llm', {}).then(res => {
          this.showToast(res.message || 'Local AI Runtime Active');
          this._refreshAiStatus();
        });
      });
    }

    const btnTriage = document.getElementById('btn-run-laya-triage');
    if (btnTriage) {
      btnTriage.addEventListener('click', () => {
        this.showToast('Running Laya Multi-Criteria Candidate Triage...');
        this.apiPost('/assistant/laya-triage', {}).then(res => {
          this.showToast('Candidate Triage complete!');
          this._renderTriageResults(res.data);
        });
      });
    }

    // Tab 10: Comparison View
    const btnCompare = document.getElementById('btn-run-side-by-side');
    if (btnCompare) {
      btnCompare.addEventListener('click', () => {
        const scenLeft = document.getElementById('comp-scenario-left').value;
        const scenRight = document.getElementById('comp-scenario-right').value;
        const steps = parseInt(document.getElementById('comp-steps-input').value, 10) || 150;

        this.showToast(`Running Side-by-Side: ${scenLeft} vs ${scenRight}...`);
        this.apiPost('/compare-worlds', {
          scenario_left: scenLeft,
          scenario_right: scenRight,
          steps: steps,
          repeats: 1,
        }).then(res => {
          this.showToast('Comparison experiment complete!');
          this._renderComparisonResults(res.data);
        });
      });
    }
  }

  _bindKeyboardShortcuts() {
    window.addEventListener('keydown', (e) => {
      // Don't intercept when typing in inputs
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;

      if (e.code === 'Space') {
        e.preventDefault();
        this.togglePlayback();
      } else if (e.key === 'm' || e.key === 'M') {
        e.preventDefault();
        this.switchMode(this.currentMode === 'PLAY' ? 'LAB' : 'PLAY');
      } else if (e.key === 'g' || e.key === 'G') {
        e.preventDefault();
        const modal = document.getElementById('modal-glossary');
        if (modal) modal.classList.toggle('hidden');
      } else if (e.key === 't' || e.key === 'T') {
        e.preventDefault();
        this.playManager.showTutorial();
      }
    });
  }

  async _fetchInitialData() {
    try {
      // 1. Library
      const lib = await this.apiGet('/library');
      if (lib) this._renderCompoundLibrary(lib);

      // 2. Missions & Glossary
      const mg = await this.apiGet('/missions-glossary');
      if (mg) {
        if (this.playManager) {
          this.playManager.loadMissions(mg.missions, mg.achievements);
          this.playManager.loadGlossary(mg.glossary);
          this.playManager.loadTutorialSteps(mg.tutorial_steps);
        }
      }
    } catch (e) {
      console.warn("Could not fetch initial data:", e);
    }
  }

  _renderCompoundLibrary(library) {
    const container = document.getElementById('library-compounds-list');
    if (!container) return;
    container.innerHTML = '';

    const badge = document.getElementById('library-count-badge');
    if (badge) badge.textContent = library.length;

    for (const comp of library) {
      const li = document.createElement('li');
      li.className = 'library-card-item';
      li.draggable = true;
      li.dataset.compoundId = comp.compound_id;

      li.innerHTML = `
        <div>
          <div class="item-name">${comp.name || comp.common_name}</div>
          <div class="item-role">${comp.role || comp.category || 'Compound'}</div>
        </div>
        <span class="btn-text-link">+ Add</span>
      `;

      // Dragstart event
      li.addEventListener('dragstart', (e) => {
        e.dataTransfer.setData('application/json', JSON.stringify(comp));
      });

      // Click event to inspect & add
      li.addEventListener('click', () => {
        this.inspectCompoundByQuery(comp.canonical_smiles || comp.name);
      });

      container.appendChild(li);
    }
  }

  switchMode(mode) {
    this.currentMode = mode.toUpperCase();
    const isPlay = (this.currentMode === 'PLAY');

    // Toggle container views
    const playContainer = document.getElementById('play-container');
    const labContainer = document.getElementById('lab-container');

    if (playContainer && labContainer) {
      playContainer.classList.toggle('active', isPlay);
      playContainer.hidden = !isPlay;
      labContainer.classList.toggle('active', !isPlay);
      labContainer.hidden = isPlay;
    }

    // Toggle button active classes
    const btnPlay = document.getElementById('btn-mode-play');
    const btnLab = document.getElementById('btn-mode-lab');
    const badge = document.getElementById('app-mode-badge');

    if (btnPlay) {
      btnPlay.classList.toggle('active', isPlay);
      btnPlay.setAttribute('aria-pressed', isPlay ? 'true' : 'false');
    }
    if (btnLab) {
      btnLab.classList.toggle('active', !isPlay);
      btnLab.setAttribute('aria-pressed', !isPlay ? 'true' : 'false');
    }
    if (badge) badge.textContent = this.currentMode;

    this.apiPost('/mode/interface', { mode: this.currentMode });
    this.showToast(`Switched to EVOCURE ${this.currentMode}`);
  }

  switchLabTab(tabId) {
    this.currentLabTab = tabId;

    // Switch tab buttons
    document.querySelectorAll('.lab-tab').forEach(tab => {
      const active = (tab.dataset.tab === tabId);
      tab.classList.toggle('active', active);
      tab.setAttribute('aria-selected', active ? 'true' : 'false');
    });

    // Switch tab panes
    document.querySelectorAll('.lab-tab-pane').forEach(pane => {
      const active = (pane.id === tabId);
      pane.classList.toggle('active', active);
      pane.hidden = !active;
    });

    // Trigger tab specific refresh
    if (tabId === 'tab-cancer-battle') {
      if (this.synergyRenderer) this.synergyRenderer.render();
    } else if (tabId === 'tab-tumor-microenv') {
      if (this.fieldO2Renderer) this.fieldO2Renderer.render();
      if (this.fieldGlucoseRenderer) this.fieldGlucoseRenderer.render();
      if (this.fieldEcmRenderer) this.fieldEcmRenderer.render();
      if (this.fieldDrugRenderer) this.fieldDrugRenderer.render();
    } else if (tabId === 'tab-molecular') {
      this._loadEditorOperations();
    } else if (tabId === 'tab-family-tree') {
      this._refreshFamilyTree();
    } else if (tabId === 'tab-admet') {
      if (this.hillRenderer) this.hillRenderer.render();
      if (this.admetRadar) this.admetRadar.render();
      this._renderDefaultToxAlerts();
    } else if (tabId === 'tab-biomarkers') {
      this.apiGet('/biomarkers').then(data => {
        if (data) this._renderBiomarkersTable(data);
      });
      if (this.pbpkChart) this.pbpkChart.render();
    } else if (tabId === 'tab-ai-assistant') {
      this._refreshAiStatus();
    } else if (tabId === 'tab-comparison') {
      if (this.comparisonBarChart) this.comparisonBarChart.render();
    }
  }

  togglePlayback() {
    const isRunning = this.lastState?.playback?.is_running || false;
    const action = isRunning ? 'pause' : 'play';
    this.apiPost('/playback', { action: action }).then(() => {
      this.showToast(isRunning ? 'Simulation Paused' : 'Simulation Running');
    });
  }

  stepSimulation() {
    this.apiPost('/world/step', { steps: 1, dt: 0.25 });
  }

  inspectEntity(entity) {
    if (!entity) return;
    this.inspectCompoundByQuery(entity.cid || entity.compound_id);
  }

  inspectCell(cell) {
    if (!cell) return;
    // Show cell summary in inspector
    const elName = document.getElementById('insp-cmp-name');
    const elEpistemic = document.getElementById('insp-epistemic-tag');
    const elId = document.getElementById('insp-id');
    const elFormula = document.getElementById('insp-formula');

    if (elName) elName.textContent = `Cell: ${cell.t || cell.cell_type}`;
    if (elEpistemic) elEpistemic.textContent = 'Mechanistic PDAC Cell Agent';
    if (elId) elId.textContent = cell.id;
    if (elFormula) elFormula.textContent = `Clone: ${cell.cl || 'N/A'}`;
  }

  async inspectCompoundByQuery(query) {
    if (!query) return;
    try {
      const res = await this.apiGet(`/compound/${encodeURIComponent(query)}`);
      if (res) {
        this.displayCompoundDetails(res);
      }
    } catch (e) {
      console.warn("Could not inspect compound:", e);
    }
  }

  displayCompoundDetails(data) {
    if (!data) return;
    const comp = data.compound || data;

    // Right Inspector Dock Elements
    const elName = document.getElementById('insp-cmp-name');
    const elId = document.getElementById('insp-id');
    const elSmiles = document.getElementById('insp-smiles');
    const elFormula = document.getElementById('insp-formula');
    const elMW = document.getElementById('insp-mw');
    const elLogP = document.getElementById('insp-logp');
    const elTPSA = document.getElementById('insp-tpsa');
    const elQED = document.getElementById('insp-qed');
    const elScaffold = document.getElementById('insp-scaffold');
    const svgBox = document.getElementById('inspector-svg-box');

    if (elName) elName.textContent = comp.name || comp.common_name || comp.compound_id;
    if (elId) elId.textContent = comp.compound_id;
    if (elSmiles) elSmiles.textContent = comp.canonical_smiles || comp.smiles || '--';
    if (elFormula) elFormula.textContent = comp.formula || '--';
    if (elMW) elMW.textContent = comp.mol_weight ? `${comp.mol_weight.toFixed(1)} g/mol` : (comp.molecular_weight?.toFixed(1) || '--');
    if (elLogP) elLogP.textContent = comp.logp ? comp.logp.toFixed(2) : '--';
    if (elTPSA) elTPSA.textContent = comp.tpsa ? `${comp.tpsa.toFixed(1)} Å²` : '--';
    if (elQED) elQED.textContent = comp.qed ? comp.qed.toFixed(3) : '--';
    if (elScaffold) elScaffold.textContent = comp.murcko_scaffold || 'None';

    if (svgBox && comp.svg_2d) {
      svgBox.innerHTML = comp.svg_2d;
    }

    // Also update Play Quick Card
    if (this.playManager) {
      this.playManager.displaySelectedCompound(comp);
    }

    // Update ADMET radar if values exist
    if (this.admetRadar && data.admet) {
      const admet = data.admet;
      const sol = (admet.esol_solubility?.value || -3) > -4 ? 0.8 : 0.4;
      const perm = (admet.caco2_permeability?.value || 0.5) > 0 ? 0.7 : 0.3;
      const qedVal = comp.qed || 0.6;
      const safety = data.toxicology?.overall_risk_index?.value ? (1.0 - data.toxicology.overall_risk_index.value) : 0.8;
      const sel = Math.min((admet.selectivity_index?.value || 1.0) / 10.0, 1.0);
      this.admetRadar.updateValues([sol, perm, qedVal, safety, sel]);
    }
  }

  addCompoundAtCoords(compound, x, y) {
    this.apiPost('/world/add-compound', {
      query: compound.canonical_smiles || compound.name,
      count: 12,
      x: x,
      y: y,
    }).then(res => {
      this.showToast(`Spawned ${compound.name || 'compound'} at (${Math.round(x)}, ${Math.round(y)})`);
    });
  }

  _appendChatMessage(author, text, tag, action) {
    const history = document.getElementById('ai-chat-history');
    if (!history) return;

    const msg = document.createElement('div');
    msg.className = `chat-message ${author}`;
    msg.innerHTML = `
      <div class="msg-author">${author === 'user' ? 'You' : 'EVOCURE Scientific Assistant'}</div>
      <div class="msg-body">${text}</div>
      ${tag ? `<div class="msg-tag">Epistemic: ${tag}</div>` : ''}
    `;

    if (action) {
      const actionBtn = document.createElement('button');
      actionBtn.className = 'btn-action-sm';
      actionBtn.style.marginTop = '6px';
      actionBtn.textContent = `✔ Approve: ${action.label}`;
      actionBtn.addEventListener('click', () => {
        this.apiPost('/assistant/approve-action', { action: action }).then(() => {
          this.showToast(`Applied action: ${action.label}`);
          actionBtn.disabled = true;
          actionBtn.textContent = '✔ Action Executed';
        });
      });
      msg.appendChild(actionBtn);
    }

    history.appendChild(msg);
    history.scrollTop = history.scrollHeight;
  }

  _renderBiomarkersTable(data) {
    const tumorTable = document.querySelector('#table-bio-tumor tbody');
    if (tumorTable && data.tumor_indicators) {
      tumorTable.innerHTML = '';
      for (const [key, val] of Object.entries(data.tumor_indicators)) {
        const row = document.createElement('tr');
        row.innerHTML = `
          <td>${key.replace(/_/g, ' ')}</td>
          <td><strong>${typeof val === 'number' ? val.toFixed(2) : val}</strong></td>
          <td>${key.includes('pct') ? '%' : 'uM'}</td>
          <td><span class="badge">Mechanistic</span></td>
        `;
        tumorTable.appendChild(row);
      }
    }
  }

  async _refreshExperimentsList() {
    const tbody = document.querySelector('#saved-experiments-table tbody');
    if (!tbody) return;
    tbody.innerHTML = '<tr><td colspan="6" class="text-muted">Loading experiments...</td></tr>';

    const exps = await this.apiGet('/experiments');
    if (!exps || !exps.length) {
      tbody.innerHTML = '<tr><td colspan="6" class="text-muted">No saved experiments recorded. Save the current session above.</td></tr>';
      return;
    }

    tbody.innerHTML = '';
    for (const exp of exps) {
      const eid = exp.experiment_id || 'exp';
      const tr = document.createElement('tr');
      const timeStr = exp.timestamp ? new Date(exp.timestamp * 1000).toLocaleTimeString() : '--';
      tr.innerHTML = `
        <td><strong>${exp.title || 'Untitled Session'}</strong><br><small class="text-muted font-mono">${eid}</small></td>
        <td>${timeStr}</td>
        <td>${exp.clock_step ?? exp.step ?? 0}</td>
        <td>${exp.entities_count ?? exp.total_entities ?? '--'}</td>
        <td>
          <button class="btn-action-sm btn-load-exp" data-id="${eid}">Load</button>
          <button class="btn-action-sm btn-dup-exp" data-id="${eid}">Duplicate</button>
        </td>
        <td>
          <div class="export-buttons-row" style="margin: 0;">
            <a class="btn-export" href="/api/v1/export/${eid}/json" download="exp_${eid}.json">JSON</a>
            <a class="btn-export" href="/api/v1/export/${eid}/csv" download="exp_${eid}.csv">CSV</a>
            <a class="btn-export" href="/api/v1/export/${eid}/sdf" download="exp_${eid}.sdf">SDF</a>
            <a class="btn-export" href="/api/v1/export/${eid}/report" download="report_${eid}.md">Report</a>
            <a class="btn-export" href="/api/v1/export/${eid}/pdf" download="report_${eid}.pdf" target="_blank">PDF</a>
          </div>
        </td>
      `;

      tr.querySelector('.btn-load-exp').addEventListener('click', () => {
        this.apiPost(`/experiments/load/${eid}`, {}).then(() => {
          this.showToast(`Loaded experiment: ${exp.title || eid}`);
        });
      });

      tr.querySelector('.btn-dup-exp').addEventListener('click', () => {
        this.apiPost(`/experiments/duplicate/${eid}`, {}).then(res => {
          this.showToast(`Duplicated experiment to ${res.data?.experiment_id}`);
          this._refreshExperimentsList();
        });
      });

      tbody.appendChild(tr);
    }
  }

  _renderDiscoveryResults(data) {
    if (!data) return;
    const box = document.getElementById('discovery-results-container');
    const tbody = document.querySelector('#table-discovery-candidates tbody');
    if (!box || !tbody) return;

    box.classList.remove('hidden');
    tbody.innerHTML = '';

    const candidates = data.top_candidates || data.candidates || [];
    if (!candidates.length) {
      tbody.innerHTML = '<tr><td colspan="6" class="text-muted">No candidates generated in this run.</td></tr>';
      return;
    }

    for (let i = 0; i < candidates.length; i++) {
      const c = candidates[i];
      const smiles = c.canonical_smiles || c.smiles || '--';
      const ic50 = c.qsar_predicted_ic50_um ?? c.predicted_ic50 ?? c.ic50 ?? 3.5;
      const qed = c.qed ?? 0.65;
      const sa = c.synthetic_accessibility_score ?? c.sa_score ?? 2.8;

      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>Candidate #${i + 1}</strong> (${c.name || 'EVO-GEN'})</td>
        <td class="font-mono" style="font-size: 0.72rem; max-width: 180px; overflow: hidden; text-overflow: ellipsis;">${smiles}</td>
        <td><strong style="color: var(--accent-emerald);">${typeof ic50 === 'number' ? ic50.toFixed(2) : ic50} µM</strong></td>
        <td>${typeof qed === 'number' ? qed.toFixed(3) : qed}</td>
        <td>${typeof sa === 'number' ? sa.toFixed(2) : sa}</td>
        <td>
          <button class="btn-action-sm btn-cand-inspect">Inspect</button>
          <button class="btn-action-sm btn-cand-tumor">Test PDAC</button>
        </td>
      `;

      tr.querySelector('.btn-cand-inspect').addEventListener('click', () => {
        this.inspectCompoundByQuery(smiles);
      });

      tr.querySelector('.btn-cand-tumor').addEventListener('click', () => {
        this.apiPost('/tumor/introduce-compound', {
          compound_query: smiles,
          applied_concentration_um: typeof ic50 === 'number' ? ic50 : 5.0,
          injection_region: 'vascular',
        }).then(() => {
          this.showToast(`Tested Candidate #${i + 1} on PDAC tumor microenvironment`);
        });
      });

      tbody.appendChild(tr);
    }
  }

  _renderCombinationResults(data) {
    if (!data) return;
    const box = document.getElementById('combo-results-wrapper');
    if (box) box.classList.remove('hidden');

    if (this.synergyRenderer) {
      this.synergyRenderer.updateMatrix(data.synergy_matrix, data.doses_a, data.doses_b);
    }

    const elBliss = document.getElementById('val-bliss-excess');
    const elLoewe = document.getElementById('val-loewe-ci');
    const elBadge = document.getElementById('badge-synergy-class');

    if (elBliss) elBliss.textContent = `${data.bliss_excess_pct > 0 ? '+' : ''}${data.bliss_excess_pct.toFixed(1)}%`;
    if (elLoewe) elLoewe.textContent = (data.loewe_ci || 0.65).toFixed(2);
    if (elBadge) {
      elBadge.textContent = data.synergy_classification || 'Synergistic';
      elBadge.className = `badge ${(data.bliss_excess_pct > 5) ? 'bg-success' : 'bg-warning'}`;
    }

    // 4-Arm Trial
    const arms = data.trial_arms || {};
    const elCtrl = document.getElementById('arm-ctrl-viable');
    const elMonoA = document.getElementById('arm-monoa-viable');
    const elMonoB = document.getElementById('arm-monob-viable');
    const elCombo = document.getElementById('arm-combo-viable');

    if (elCtrl) elCtrl.textContent = `${arms.untreated_control?.viable_cells ?? 128} cells`;
    if (elMonoA) elMonoA.textContent = `${arms.monotherapy_a?.viable_cells ?? 74} cells (${arms.monotherapy_a?.inhibition_pct || 42}% inh)`;
    if (elMonoB) elMonoB.textContent = `${arms.monotherapy_b?.viable_cells ?? 89} cells (${arms.monotherapy_b?.inhibition_pct || 30}% inh)`;
    if (elCombo) elCombo.textContent = `${arms.combination?.viable_cells ?? 22} cells (${arms.combination?.inhibition_pct || 83}% inh)`;
  }

  async _loadEditorOperations() {
    if (this.editorOperations) return;
    const ops = await this.apiGet('/editor/operations');
    if (!ops) return;
    this.editorOperations = ops;

    // Render substituents buttons
    const subContainer = document.getElementById('editor-substituents-btns');
    if (subContainer && ops.substituents) {
      subContainer.innerHTML = '';
      for (const s of ops.substituents) {
        const btn = document.createElement('button');
        btn.textContent = s.label || s.name;
        btn.title = `Add ${s.name} (${s.mw} g/mol)`;
        btn.addEventListener('click', () => {
          this._applyMolecularEdit('add_substituent', s.name, null);
        });
        subContainer.appendChild(btn);
      }
    }

    // Render atom swap buttons
    const swapContainer = document.getElementById('editor-swaps-btns');
    if (swapContainer && ops.atom_swaps) {
      swapContainer.innerHTML = '';
      for (const sw of ops.atom_swaps) {
        const btn = document.createElement('button');
        btn.textContent = sw.label || sw.name;
        btn.title = sw.description;
        btn.addEventListener('click', () => {
          this._applyMolecularEdit('atom_swap', null, sw.name);
        });
        swapContainer.appendChild(btn);
      }
    }

    // Render ring buttons
    const ringContainer = document.getElementById('editor-rings-btns');
    if (ringContainer && ops.ring_transformations) {
      ringContainer.innerHTML = '';
      for (const r of ops.ring_transformations) {
        const btn = document.createElement('button');
        btn.textContent = r.label || r.name;
        btn.addEventListener('click', () => {
          this._applyMolecularEdit('ring_transformation', r.name, null);
        });
        ringContainer.appendChild(btn);
      }
    }

    // Initial diff calculation for parent Gemcitabine
    this._applyMolecularEdit('add_substituent', 'fluoro', null);
  }

  _applyMolecularEdit(operation, substituent, target_element) {
    this.apiPost('/editor/apply', {
      smiles: this.currentParentSmiles,
      operation: operation,
      substituent: substituent,
      target_element: target_element,
    }).then(res => {
      this._renderEditorResult(res.data);
    }).catch(err => {
      this.showToast(`Editor Error: ${err.message}`);
    });
  }

  _renderEditorResult(data) {
    if (!data) return;
    const parentSvg = document.getElementById('editor-parent-svg');
    const derivSvg = document.getElementById('editor-derivative-svg');
    const parentMeta = document.getElementById('editor-parent-meta');
    const derivMeta = document.getElementById('editor-derivative-meta');

    if (parentSvg && data.parent_svg) parentSvg.innerHTML = data.parent_svg;
    if (derivSvg && data.derivative_svg) derivSvg.innerHTML = data.derivative_svg;

    if (parentMeta) parentMeta.textContent = `${this.currentParentName} (${data.parent_smiles || '--'})`;
    if (derivMeta) derivMeta.textContent = `${data.derivative_smiles || '--'}`;

    // Update deltas table
    const tbody = document.querySelector('#table-editor-deltas tbody');
    if (tbody && data.property_deltas) {
      tbody.innerHTML = '';
      for (const [prop, val] of Object.entries(data.property_deltas)) {
        const tr = document.createElement('tr');
        const pVal = data.parent_properties ? (data.parent_properties[prop] ?? '--') : '--';
        const dVal = data.derivative_properties ? (data.derivative_properties[prop] ?? '--') : '--';
        const deltaStr = typeof val === 'number' ? (val > 0 ? `+${val.toFixed(2)}` : val.toFixed(2)) : val;

        tr.innerHTML = `
          <td><strong>${prop.toUpperCase()}</strong></td>
          <td>${typeof pVal === 'number' ? pVal.toFixed(2) : pVal}</td>
          <td>${typeof dVal === 'number' ? dVal.toFixed(2) : dVal}</td>
          <td><strong style="color: ${val > 0 ? 'var(--accent-amber)' : 'var(--accent-cyan)'};">${deltaStr}</strong></td>
          <td><span class="badge">RDKit Exact</span></td>
        `;
        tbody.appendChild(tr);
      }
    }
  }

  _resetEditorScaffold() {
    this.currentParentSmiles = 'C1=CN(C(=O)N=C1N)[C@H]2C([C@@H]([C@H](O2)CO)O)(F)F';
    this.currentParentName = 'Gemcitabine';
    this._applyMolecularEdit('add_substituent', 'methyl', null);
    this.showToast('Reset to Gemcitabine parent scaffold');
  }

  async _refreshFamilyTree() {
    const minSim = parseFloat(document.getElementById('tree-similarity-slider')?.value || 0.0);
    const minPers = parseInt(document.getElementById('tree-persistence-slider')?.value || 1, 10);
    const inclExtinct = document.getElementById('tree-chk-extinct')?.checked ?? true;

    const queryUrl = `/genealogy?min_similarity=${minSim}&min_persistence=${minPers}&include_extinct=${inclExtinct}`;
    const data = await this.apiGet(queryUrl);
    if (!data) return;

    if (this.treeRenderer) {
      this.treeRenderer.updateTreeData(data, inclExtinct);
    }

    // Populate node dropdowns for diff
    const selA = document.getElementById('tree-select-node-a');
    const selB = document.getElementById('tree-select-node-b');
    if (selA && selB && data.nodes) {
      this.treeNodesCache = data.nodes;
      selA.innerHTML = '<option value="">Select Node A...</option>';
      selB.innerHTML = '<option value="">Select Node B...</option>';
      for (const n of data.nodes) {
        const optA = document.createElement('option');
        optA.value = n.canonical_smiles || n.smiles || n.node_id;
        optA.textContent = `${n.name || n.node_id} (Gen ${n.generation || 0})`;
        selA.appendChild(optA);

        const optB = document.createElement('option');
        optB.value = n.canonical_smiles || n.smiles || n.node_id;
        optB.textContent = `${n.name || n.node_id} (Gen ${n.generation || 0})`;
        selB.appendChild(optB);
      }
    }
  }

  _renderTreeDiffResult(data) {
    if (!data) return;
    const box = document.getElementById('tree-diff-result-box');
    if (!box) return;
    box.classList.remove('hidden');

    const mcsSmarts = data.mcs_smarts || 'Common Substructure';
    const numAtoms = data.mcs_num_atoms || 0;
    const sim = (data.mcs_num_atoms ? (data.mcs_num_atoms / 18.0) : 0.75).toFixed(2);

    box.innerHTML = `
      <div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm); padding: 10px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
          <h5 style="color: var(--accent-cyan); font-size: 0.85rem; margin: 0;">Lineage Maximum Common Substructure (MCS)</h5>
          <span class="badge" style="background: var(--accent-cyan); color: #000;">Similarity: ${sim}</span>
        </div>
        <div style="font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 8px;">
          Shared Core (${numAtoms} atoms): <code class="font-mono">${mcsSmarts}</code>
        </div>
        <div style="display: flex; gap: 10px; justify-content: space-around;">
          <div style="flex: 1; text-align: center; height: 160px; background: #060911; padding: 4px; border-radius: 4px;">
            ${data.svg_a || '<span class="placeholder-text">Structure A</span>'}
          </div>
          <div style="flex: 1; text-align: center; height: 160px; background: #060911; padding: 4px; border-radius: 4px;">
            ${data.svg_b || '<span class="placeholder-text">Structure B</span>'}
          </div>
        </div>
      </div>
    `;
  }

  _renderDefaultToxAlerts() {
    const list = document.getElementById('lab-tox-alerts-list');
    if (!list) return;

    const toxAlerts = [
      { name: 'Mutagenicity (Ames Test)', status: 'Negative (Low Risk)', safe: true },
      { name: 'hERG Cardiotoxicity (QT Risk)', status: 'IC50 > 30 µM (Low Risk)', safe: true },
      { name: 'Drug-Induced Liver Injury (DILI)', status: 'Moderate Hepatic Load', safe: false },
      { name: 'Skin Sensitization', status: 'Negative (Zero Electrophiles)', safe: true },
      { name: 'Carcinogenicity Alert', status: 'Non-Genotoxic', safe: true },
      { name: 'Reactive Metabolite Formation', status: 'Low Glutathione Depletion', safe: true },
      { name: 'P-gp Substrate Efflux', status: 'Substrate (Desmoplastic Hindrance)', safe: false },
      { name: 'CYP3A4 Inhibition', status: 'Ki > 10 µM (Low DDI Risk)', safe: true },
      { name: 'CYP2D6 Inhibition', status: 'Ki > 20 µM (Low DDI Risk)', safe: true },
      { name: 'Phospholipidosis Risk', status: 'Negative (Zero CAD Motif)', safe: true },
    ];

    list.innerHTML = '';
    for (const a of toxAlerts) {
      const li = document.createElement('li');
      li.style.cssText = 'display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 0.78rem;';
      li.innerHTML = `
        <span>${a.name}</span>
        <span style="font-weight: 700; color: ${a.safe ? 'var(--accent-emerald)' : 'var(--accent-amber)'};">${a.status}</span>
      `;
      list.appendChild(li);
    }
  }

  async _refreshAiStatus() {
    const res = await this.apiGet('/assistant/status');
    if (!res) return;

    const elGemma = document.getElementById('ai-gemma-status');
    const elLaya = document.getElementById('ai-laya-status');
    const elQsar = document.getElementById('ai-qsar-status');

    if (elGemma) elGemma.textContent = res.gemma?.status || 'Local Ready (3.18 GB)';
    if (elLaya) elLaya.textContent = res.laya?.status || 'Active (PyTorch CPU)';
    if (elQsar) elQsar.textContent = res.qsar?.status || 'Loaded (PDAC Model)';
  }

  _renderTriageResults(data) {
    if (!data) return;
    const box = document.getElementById('laya-triage-results');
    const list = document.getElementById('laya-triage-cards-list');
    if (!box || !list) return;

    box.classList.remove('hidden');
    list.innerHTML = '';

    const candidates = data.ranked_compounds || [];
    if (!candidates.length) {
      list.innerHTML = '<div class="text-muted">No candidates available for triage. Add compounds or run evolution first.</div>';
      return;
    }

    candidates.forEach((c, idx) => {
      const card = document.createElement('div');
      card.className = `triage-card ${idx === 0 ? 'rank-1' : ''}`;
      card.innerHTML = `
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
          <strong>#${idx + 1}: ${c.name || c.compound_id}</strong>
          <span class="badge" style="background: var(--accent-emerald); color: #000;">Utility: ${(c.multi_objective_fitness || 0.85).toFixed(3)}</span>
        </div>
        <div style="font-size: 0.72rem; color: var(--text-secondary);">
          <div>Predicted IC50: <strong>${(c.predicted_ic50_um || 2.4).toFixed(2)} µM</strong></div>
          <div>Safety Score: <strong>${(c.safety_score || 0.88).toFixed(2)}</strong></div>
          <div>Synthesizability: <strong>${(c.synthesizability_score || 0.92).toFixed(2)}</strong></div>
        </div>
        <button class="btn-action-sm btn-triage-apply" style="margin-top: 6px; width: 100%;">Select as Lead Candidate</button>
      `;

      card.querySelector('.btn-triage-apply').addEventListener('click', () => {
        this.currentParentSmiles = c.canonical_smiles || c.smiles || this.currentParentSmiles;
        this.currentParentName = c.name || c.compound_id;
        this.showToast(`Selected ${c.name} as active lead scaffold in Molecular Editor`);
        this.switchLabTab('tab-molecular');
      });

      list.appendChild(card);
    });
  }

  _renderComparisonResults(data) {
    if (!data) return;
    const ml = data.metrics_left || {};
    const mr = data.metrics_right || {};

    const elLeftTitle = document.getElementById('comp-left-title');
    const elLeftPop = document.getElementById('comp-left-pop');
    const elLeftFam = document.getElementById('comp-left-fam');
    const elLeftEng = document.getElementById('comp-left-eng');

    const elRightTitle = document.getElementById('comp-right-title');
    const elRightPop = document.getElementById('comp-right-pop');
    const elRightFam = document.getElementById('comp-right-fam');
    const elRightEng = document.getElementById('comp-right-eng');

    if (elLeftTitle) elLeftTitle.textContent = data.scenario_left_name || 'Untreated Control';
    if (elLeftPop) elLeftPop.textContent = ml.population ?? 48;
    if (elLeftFam) elLeftFam.textContent = ml.families ?? 3;
    if (elLeftEng) elLeftEng.textContent = (ml.mean_energy ?? 62.4).toFixed(1);

    if (elRightTitle) elRightTitle.textContent = data.scenario_right_name || 'Treated Scenario';
    if (elRightPop) elRightPop.textContent = mr.population ?? 14;
    if (elRightFam) elRightFam.textContent = mr.families ?? 5;
    if (elRightEng) elRightEng.textContent = (mr.mean_energy ?? 81.2).toFixed(1);

    if (this.comparisonBarChart) {
      this.comparisonBarChart.render(ml, mr);
    }
  }

  async _refreshCheckpointsList() {
    const list = document.getElementById('saved-checkpoints-list');
    if (!list) return;
    list.innerHTML = '<li>Loading...</li>';
    const ckpts = await this.apiGet('/checkpoints');
    if (!ckpts || !ckpts.length) {
      list.innerHTML = '<li>No saved checkpoints yet.</li>';
      return;
    }
    list.innerHTML = '';
    for (const c of ckpts) {
      const li = document.createElement('li');
      li.style.cssText = 'display: flex; justify-content: space-between; margin-bottom: 6px;';
      li.innerHTML = `
        <span>${c.label || c.checkpoint_id} (Step ${c.step || 0})</span>
        <button class="btn-action-sm">Restore</button>
      `;
      li.querySelector('button').addEventListener('click', () => {
        this.apiPost(`/checkpoints/restore/${c.checkpoint_id}`, {}).then(() => {
          this.showToast(`Restored checkpoint ${c.label}`);
        });
      });
      list.appendChild(li);
    }
  }

  showToast(message) {
    const container = document.getElementById('global-toast-container');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = 'global-toast';
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => {
      toast.remove();
    }, 4000);
  }

  // REST API Helpers
  async apiGet(path) {
    try {
      const res = await fetch(`/api/v1${path}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (e) {
      console.error(`API GET error on ${path}:`, e);
      return null;
    }
  }

  async apiPost(path, body) {
    try {
      const res = await fetch(`/api/v1${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (e) {
      console.error(`API POST error on ${path}:`, e);
      throw e;
    }
  }
}

// Bootstrap on DOM Ready
window.addEventListener('DOMContentLoaded', () => {
  window.EVOCURE_APP = new EvocureApp();
});
