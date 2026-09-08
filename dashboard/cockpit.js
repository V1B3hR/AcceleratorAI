/**
 * Cockpit Controller
 * Manages analog round dials, live telemetry updates, simulation loop,
 * and user interactions (Chaos Shock kick, slider adjustments).
 */

class AnalogGauge {
  constructor(canvasId, config) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.minVal = config.minVal || 0;
    this.maxVal = config.maxVal || 100;
    this.units = config.units || '';
    this.accentColor = config.color || '#00f0ff';
    this.dangerVal = config.dangerVal || null;
    this.currentVal = this.minVal;
    this.targetVal = this.minVal;
  }

  set(val) {
    this.targetVal = Math.min(this.maxVal, Math.max(this.minVal, val));
  }

  draw() {
    // Smooth needle damping
    this.currentVal += (this.targetVal - this.currentVal) * 0.12;

    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    const cx = w / 2;
    const cy = h / 2;
    const radius = w * 0.42;

    ctx.clearRect(0, 0, w, h);

    // Arc sweep from 135 deg to 405 deg (270 deg total)
    const startAngle = 0.75 * Math.PI;
    const endAngle = 2.25 * Math.PI;
    const totalSweep = endAngle - startAngle;

    // Background track
    ctx.beginPath();
    ctx.arc(cx, cy, radius, startAngle, endAngle);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 10;
    ctx.lineCap = 'round';
    ctx.stroke();

    // Active value arc
    const pct = (this.currentVal - this.minVal) / (this.maxVal - this.minVal);
    const curAngle = startAngle + (pct * totalSweep);

    ctx.beginPath();
    ctx.arc(cx, cy, radius, startAngle, curAngle);
    ctx.strokeStyle = this.accentColor;
    ctx.lineWidth = 10;
    ctx.lineCap = 'round';
    ctx.shadowColor = this.accentColor;
    ctx.shadowBlur = 12;
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Ticks
    const numTicks = 9;
    for (let i = 0; i < numTicks; i++) {
      const a = startAngle + (i / (numTicks - 1)) * totalSweep;
      const r1 = radius - 8;
      const r2 = radius - 16;
      ctx.beginPath();
      ctx.moveTo(cx + Math.cos(a) * r1, cy + Math.sin(a) * r1);
      ctx.lineTo(cx + Math.cos(a) * r2, cy + Math.sin(a) * r2);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Needle
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(curAngle);

    ctx.beginPath();
    ctx.moveTo(-4, 0);
    ctx.lineTo(radius - 10, 0);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 3;
    ctx.shadowColor = '#ffffff';
    ctx.shadowBlur = 8;
    ctx.stroke();

    // Center pivot
    ctx.beginPath();
    ctx.arc(0, 0, 7, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
    ctx.restore();
  }
}

document.addEventListener('DOMContentLoaded', () => {
  // Initialize Visualizers
  const turbineVis = new TurbineVisualizer('turbineCanvas');
  const oscVis = new OscilloscopeVisualizer('oscilloscopeCanvas');
  const dnaVis = new DNAHelixVisualizer('dnaHelixCanvas');

  // Initialize Analog Gauges
  const gaugeRpm = new AnalogGauge('gaugeRpmCanvas', { minVal: 0, maxVal: 7000, color: '#00f0ff' });
  const gaugeBoost = new AnalogGauge('gaugeBoostCanvas', { minVal: 0, maxVal: 28, color: '#00f0ff' });
  const gaugeTemp = new AnalogGauge('gaugeTempCanvas', { minVal: 200, maxVal: 950, color: '#ff9d00', dangerVal: 800 });
  const gaugeTorque = new AnalogGauge('gaugeTorqueCanvas', { minVal: 0, maxVal: 60, color: '#00ff88' });

  // UI Element Handles
  const elImpellerRpm = document.getElementById('impeller-rpm-tag');
  const elOverlayBoost = document.getElementById('overlay-boost');
  const elOverlayTemp = document.getElementById('overlay-temp');

  const elDigitalRpm = document.getElementById('digital-rpm');
  const elDigitalBoost = document.getElementById('digital-boost');
  const elDigitalTemp = document.getElementById('digital-temp');
  const elDigitalTorque = document.getElementById('digital-torque');

  const elValAfr = document.getElementById('val-afr');
  const elBarAfr = document.getElementById('bar-afr');
  const elValWastegate = document.getElementById('val-wastegate');
  const elBarWastegate = document.getElementById('bar-wastegate');
  const elValLoss = document.getElementById('val-loss');
  const elBarLoss = document.getElementById('bar-loss');

  // DNA Braided Helices Handles
  const elResonanceBadge = document.getElementById('resonance-badge');
  const elValShaftEk = document.getElementById('val-shaft-ek');
  const elValPhaseTension = document.getElementById('val-phase-tension');
  const elValWindingRevs = document.getElementById('val-winding-revs');

  // VGT Multi-Port & Swirl Handles
  const elApertureHyper = document.getElementById('val-aperture-hyper');
  const elBarApertureHyper = document.getElementById('bar-aperture-hyper');
  const elVelHyper = document.getElementById('val-vel-hyper');
  const elPctHyper = document.getElementById('val-pct-hyper');

  const elApertureCruise = document.getElementById('val-aperture-cruise');
  const elBarApertureCruise = document.getElementById('bar-aperture-cruise');
  const elVelCruise = document.getElementById('val-vel-cruise');

  const elApertureSlowmo = document.getElementById('val-aperture-slowmo');
  const elBarApertureSlowmo = document.getElementById('bar-aperture-slowmo');
  const elVelSlowmo = document.getElementById('val-vel-slowmo');

  const elHomogeneityBadge = document.getElementById('homogeneity-badge');
  const elCurriculumFactor = document.getElementById('val-curriculum-factor');

  const elCycleCount = document.getElementById('cycle-count');
  const elLedSynth = document.getElementById('led-synth');
  const elLedReal = document.getElementById('led-real');
  const elLedShock = document.getElementById('led-shock');

  const elCountSynth = document.getElementById('count-synth');
  const elCountReal = document.getElementById('count-real');
  const elCountShock = document.getElementById('count-shock');

  const btnNos = document.getElementById('btn-nos-shock');
  const sliderBoost = document.getElementById('slider-boost');
  const labelBoostTarget = document.getElementById('label-boost-target');
  const sliderJitter = document.getElementById('slider-jitter');
  const labelJitter = document.getElementById('label-jitter');
  const sliderWastegate = document.getElementById('slider-wastegate');
  const labelWastegate = document.getElementById('label-wastegate');
  const telemetryLog = document.getElementById('telemetry-log');

  // Internal Engine State
  let state = {
    step: 0,
    rpm: 1200,
    boostPsi: 14.7,
    tempC: 280,
    torqueNm: 12.4,
    loss: 1.25,
    afr: 14.7,
    wastegatePct: 0.0,
    synthFired: false,
    realFired: false,
    shockFired: false,
    counts: { synth: 0, real: 0, shock: 0 },
    boostTarget: 14.7,
    jitter: 0.15,
    wastegateThreshold: 5.0,
    helicalResonance: 0.42,
    phaseTension: 0.15,
    shaftEk: 120.0,
    windingRevs: 0.0,
    hyperAperture: 0.15,
    cruiseAperture: 0.50,
    slowmoAperture: 0.85,
    hyperPct: 18.5,
    homogeneityPct: 100.0,
    curriculumFactor: 1.05,
  };

  function appendLog(msg, colorClass = 'text-cyan') {
    const line = document.createElement('div');
    line.className = `log-line ${colorClass}`;
    line.textContent = msg;
    telemetryLog.appendChild(line);
    telemetryLog.scrollTop = telemetryLog.scrollHeight;
    while (telemetryLog.childNodes.length > 25) {
      telemetryLog.removeChild(telemetryLog.firstChild);
    }
  }

  // Trigger Chaos Shock ("kopniak z boku")
  function fireChaosShock() {
    state.shockFired = true;
    state.counts.shock++;
    state.rpm = Math.min(6800, state.rpm + 2200);
    state.tempC = Math.min(880, state.tempC + 160);
    state.torqueNm = Math.min(55, state.torqueNm + 28);
    state.loss = Math.max(0.05, state.loss * 0.65); // Breakthrough!

    appendLog(`[KICK] ⚡ CHAOS SHOCK INJECTED! Non-linear perturbation applied. Local minimum shattered.`, 'text-red');

    // Notify backend if running with live server
    fetch('/api/action/nos', { method: 'POST' }).catch(() => {});

    setTimeout(() => { state.shockFired = false; }, 400);
  }

  btnNos.addEventListener('click', fireChaosShock);

  // Slider controls
  sliderBoost.addEventListener('input', (e) => {
    state.boostTarget = parseFloat(e.target.value);
    labelBoostTarget.textContent = `${state.boostTarget.toFixed(1)} PSI`;
    appendLog(`[ECU] Manifold Target Boost adjusted to ${state.boostTarget.toFixed(1)} PSI`, 'text-cyan');
    fetch('/api/action/boost', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ boost_psi: state.boostTarget }),
    }).catch(() => {});
  });

  sliderJitter.addEventListener('input', (e) => {
    state.jitter = parseFloat(e.target.value);
    labelJitter.textContent = state.jitter.toFixed(2);
  });

  sliderWastegate.addEventListener('input', (e) => {
    state.wastegateThreshold = parseFloat(e.target.value);
    labelWastegate.textContent = `${state.wastegateThreshold.toFixed(1)} Norm`;
  });

  // Telemetry Connection (SSE + REST with automatic fallback)
  function connectTelemetryStream() {
    // 1. Try Server-Sent Events (SSE)
    if (window.EventSource) {
      try {
        const evtSource = new EventSource('/api/stream');
        evtSource.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data && data.step) {
              applyLiveTelemetry(data);
            }
          } catch (e) {}
        };
        evtSource.onopen = () => {
          appendLog(`[LINK] Connected to live AcceleratorAI Python Engine via SSE!`, 'text-green');
        };
        evtSource.onerror = () => {
          evtSource.close();
          startHttpPolling();
        };
        return;
      } catch (e) {}
    }
    startHttpPolling();
  }

  function startHttpPolling() {
    setInterval(() => {
      fetch('/api/telemetry')
        .then(res => res.json())
        .then(data => {
          if (data && data.step) {
            applyLiveTelemetry(data);
          }
        })
        .catch(() => {});
    }, 150);
  }

  function applyLiveTelemetry(data) {
    state.step = data.step || state.step;
    state.rpm = data.rpm || state.rpm;
    state.boostPsi = data.boost_psi || state.boostPsi;
    state.tempC = data.pyrometer_temp_c || state.tempC;
    state.torqueNm = data.learning_torque_nm || state.torqueNm;
    state.loss = data.loss || state.loss;
    state.afr = data.air_fuel_ratio || state.afr;
    state.wastegatePct = data.wastegate_open_pct || 0.0;
    if (data.helical_resonance !== undefined) state.helicalResonance = data.helical_resonance;
    if (data.phase_tension !== undefined) state.phaseTension = data.phase_tension;
    if (data.shaft_kinetic_energy_j !== undefined) state.shaftEk = data.shaft_kinetic_energy_j;
    if (data.winding_number !== undefined) state.windingRevs = data.winding_number;
    if (data.hyper_flow_aperture !== undefined) state.hyperAperture = data.hyper_flow_aperture;
    if (data.cruise_flow_aperture !== undefined) state.cruiseAperture = data.cruise_flow_aperture;
    if (data.slowmo_flow_aperture !== undefined) state.slowmoAperture = data.slowmo_flow_aperture;
    if (data.hyper_flow_pct !== undefined) state.hyperPct = data.hyper_flow_pct;
    if (data.homogeneity_pct !== undefined) state.homogeneityPct = data.homogeneity_pct;
    if (data.curriculum_weight_mean !== undefined) state.curriculumFactor = data.curriculum_weight_mean;
  }

  connectTelemetryStream();

  // High-fidelity local physics loop for standalone demonstration
  setInterval(() => {
    state.step++;

    // Spool boost towards target
    state.boostPsi += (state.boostTarget - state.boostPsi) * 0.05;

    // Simulate asynchronous injector firings
    // Synthetic injector (~ every 8 steps)
    const synthWave = Math.sin(0.25 * state.step + 0.5 + (Math.random() - 0.5) * state.jitter);
    if (synthWave > 0.65) {
      state.synthFired = true;
      state.counts.synth++;
      setTimeout(() => { state.synthFired = false; }, 250);
    }

    // Real-world injector (~ every 14 steps)
    const realWave = Math.sin(0.18 * state.step + 1.8 + (Math.random() - 0.5) * state.jitter);
    if (realWave > 0.7) {
      state.realFired = true;
      state.counts.real++;
      setTimeout(() => { state.realFired = false; }, 300);
    }

    // Gradual loss descent with occasional mini-plateau
    state.loss = Math.max(0.04, state.loss * 0.996);
    state.tempC = 200 + (state.loss * 220) + (state.boostPsi * 8);

    // Wastegate regulation
    if (state.boostPsi > 20) {
      state.wastegatePct = ((state.boostPsi - 20) / 8) * 100;
    } else {
      state.wastegatePct = Math.max(0, state.wastegatePct * 0.85);
    }

    // RPM & Shaft Kinetic Energy dynamics
    const targetRpm = 1000 + (state.boostPsi * 120) + (state.torqueNm * 65);
    state.rpm += (targetRpm - state.rpm) * 0.1;
    const omega = (state.rpm * 2 * Math.PI) / 60;
    state.shaftEk = 0.5 * 0.08 * (omega * omega);

    // DNA Helical Resonance oscillation simulation
    state.helicalResonance = Math.sin(0.08 * state.step) * 0.6 + 0.2;
    state.phaseTension = Math.max(0.05, 1.0 - Math.max(0, state.helicalResonance));
    state.windingRevs = (state.step * 0.04).toFixed(1);

    // VGT Multi-Port Aperture Dynamics
    // When resonance is constructive (focus), hyper-port narrows; when tension, it widens
    const h = state.helicalResonance;
    state.hyperAperture = Math.max(0.06, Math.min(0.35, 0.15 - h * 0.04));
    state.slowmoAperture = Math.max(0.60, Math.min(0.95, 0.85 + h * 0.05));
    state.hyperPct = Math.max(5.0, Math.min(45.0, 20.0 + h * 12.0));
    state.curriculumFactor = Math.max(0.8, Math.min(1.4, 1.0 + (1.0 / Math.max(0.05, state.hyperAperture) - 6.67) * 0.03));

    // UI Updates
    elCycleCount.textContent = state.step;
    elImpellerRpm.textContent = `${Math.round(state.rpm)} RPM`;
    elOverlayBoost.textContent = `${state.boostPsi.toFixed(1)} PSI`;
    elOverlayTemp.textContent = `${Math.round(state.tempC)} °C`;

    elDigitalRpm.textContent = Math.round(state.rpm);
    elDigitalBoost.textContent = `${state.boostPsi.toFixed(1)} PSI`;
    elDigitalTemp.textContent = `${Math.round(state.tempC)} °C`;
    elDigitalTorque.textContent = `${state.torqueNm.toFixed(1)} Nm`;

    gaugeRpm.set(state.rpm);
    gaugeBoost.set(state.boostPsi);
    gaugeTemp.set(state.tempC);
    gaugeTorque.set(state.torqueNm);

    elValAfr.textContent = `${state.afr.toFixed(1)} : 1`;
    elBarAfr.style.width = `${Math.min(100, (state.afr / 20) * 100)}%`;

    elValWastegate.textContent = `${state.wastegatePct.toFixed(1)}%`;
    elBarWastegate.style.width = `${state.wastegatePct}%`;

    elValLoss.textContent = state.loss.toFixed(4);
    elBarLoss.style.width = `${Math.min(100, state.loss * 70)}%`;

    // DNA Braided UI Updates
    const isConstructive = state.helicalResonance >= 0.0;
    if (elResonanceBadge) {
      elResonanceBadge.className = `resonance-badge font-mono ${isConstructive ? '' : 'tension'}`;
      const sign = state.helicalResonance >= 0 ? '+' : '';
      const mode = isConstructive ? 'CONSTRUCTIVE' : 'TENSION / BIFURCATION';
      elResonanceBadge.textContent = `RESONANCE: ${sign}${state.helicalResonance.toFixed(3)} (${mode})`;
    }
    if (elValShaftEk) elValShaftEk.textContent = `${state.shaftEk.toFixed(1)} J`;
    if (elValPhaseTension) elValPhaseTension.textContent = state.phaseTension.toFixed(3);
    if (elValWindingRevs) elValWindingRevs.textContent = `${state.windingRevs} rev`;

    // VGT Multi-Port UI Updates
    if (elApertureHyper) {
      const velHyper = (1.0 / Math.max(0.01, state.hyperAperture)).toFixed(1);
      const velCruise = (1.0 / Math.max(0.01, state.cruiseAperture)).toFixed(1);
      const velSlowmo = (1.0 / Math.max(0.01, state.slowmoAperture)).toFixed(1);

      elApertureHyper.textContent = `${state.hyperAperture.toFixed(3)} ap`;
      elBarApertureHyper.style.width = `${Math.min(100, state.hyperAperture * 100)}%`;
      elVelHyper.textContent = `${velHyper}x`;
      elPctHyper.textContent = `${state.hyperPct.toFixed(1)}%`;

      elApertureCruise.textContent = `${state.cruiseAperture.toFixed(3)} ap`;
      elBarApertureCruise.style.width = `${Math.min(100, state.cruiseAperture * 100)}%`;
      elVelCruise.textContent = `${velCruise}x`;

      elApertureSlowmo.textContent = `${state.slowmoAperture.toFixed(3)} ap`;
      elBarApertureSlowmo.style.width = `${Math.min(100, state.slowmoAperture * 100)}%`;
      elVelSlowmo.textContent = `${velSlowmo}x`;

      if (elHomogeneityBadge) {
        elHomogeneityBadge.textContent = `SWIRL HOMOGENEITY: ${state.homogeneityPct.toFixed(0)}%`;
      }
      if (elCurriculumFactor) {
        elCurriculumFactor.textContent = `${state.curriculumFactor.toFixed(3)}x`;
      }
    }

    // Injector LEDs & Counts
    elLedSynth.classList.toggle('pulsing', state.synthFired);
    elLedReal.classList.toggle('pulsing', state.realFired);
    elLedShock.classList.toggle('pulsing', state.shockFired);

    elCountSynth.textContent = state.counts.synth;
    elCountReal.textContent = state.counts.real;
    elCountShock.textContent = state.counts.shock;
  }, 100);

  // 60 FPS Render Loop for Turbine, DNA Helix and Gauges
  function renderLoop() {
    turbineVis.draw(state.rpm, state.boostPsi, state.tempC);
    oscVis.draw(state.synthFired, state.realFired, state.shockFired);
    dnaVis.draw(state.helicalResonance, state.phaseTension);
    gaugeRpm.draw();
    gaugeBoost.draw();
    gaugeTemp.draw();
    gaugeTorque.draw();
    requestAnimationFrame(renderLoop);
  }
  requestAnimationFrame(renderLoop);
});
