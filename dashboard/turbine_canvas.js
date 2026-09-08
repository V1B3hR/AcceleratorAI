/**
 * Turbine Canvas Visualizer
 * Renders high-frame-rate animated turbine impeller rotor and async injector oscilloscope.
 */

class TurbineVisualizer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.angle = 0;
    this.numBlades = 14;
    this.particles = [];
    this.initParticles(35);
  }

  initParticles(count) {
    for (let i = 0; i < count; i++) {
      this.particles.push({
        x: Math.random() * this.canvas.width,
        y: Math.random() * this.canvas.height,
        speed: 1 + Math.random() * 3,
        size: 1 + Math.random() * 2,
        alpha: 0.2 + Math.random() * 0.6,
      });
    }
  }

  draw(rpm, boostPsi, tempC) {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    const cx = w / 2;
    const cy = h / 2;
    const radius = w * 0.42;

    ctx.clearRect(0, 0, w, h);

    // 1. Fluid Airflow Particles (Medium stream entering the intake)
    ctx.save();
    for (let p of this.particles) {
      p.x += p.speed * (1 + boostPsi * 0.15);
      if (p.x > w) p.x = 0;
      ctx.fillStyle = `rgba(0, 240, 255, ${p.alpha * 0.4})`;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();

    // 2. Outer Turbine Casing
    ctx.save();
    ctx.translate(cx, cy);

    // Thermal outer glow
    const thermalRatio = Math.min(1.0, Math.max(0.0, (tempC - 200) / 750));
    const glowColor = thermalRatio > 0.5 ? 'rgba(255, 60, 30, 0.45)' : 'rgba(0, 240, 255, 0.35)';

    ctx.beginPath();
    ctx.arc(0, 0, radius + 8, 0, Math.PI * 2);
    ctx.strokeStyle = glowColor;
    ctx.lineWidth = 4;
    ctx.shadowColor = glowColor;
    ctx.shadowBlur = 20;
    ctx.stroke();

    // Stator ring
    ctx.beginPath();
    ctx.arc(0, 0, radius, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.lineWidth = 6;
    ctx.stroke();

    // Stator guide vanes
    for (let i = 0; i < 8; i++) {
      const a = (i * Math.PI * 2) / 8;
      ctx.beginPath();
      ctx.moveTo(Math.cos(a) * (radius - 12), Math.sin(a) * (radius - 12));
      ctx.lineTo(Math.cos(a) * (radius + 6), Math.sin(a) * (radius + 6));
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // 3. Rotating Impeller Blades
    ctx.rotate(this.angle);

    for (let i = 0; i < this.numBlades; i++) {
      const bladeAngle = (i * Math.PI * 2) / this.numBlades;
      ctx.save();
      ctx.rotate(bladeAngle);

      // Curved aerodynamic titanium blade
      ctx.beginPath();
      ctx.moveTo(30, 0);
      ctx.bezierCurveTo(radius * 0.4, 18, radius * 0.7, -15, radius - 4, 8);
      ctx.bezierCurveTo(radius * 0.7, -5, radius * 0.4, 25, 30, 0);
      ctx.closePath();

      const bladeGrad = ctx.createLinearGradient(30, 0, radius, 0);
      if (thermalRatio > 0.6) {
        bladeGrad.addColorStop(0, '#3a1a1a');
        bladeGrad.addColorStop(0.7, '#ff5500');
        bladeGrad.addColorStop(1, '#ffdd88');
      } else {
        bladeGrad.addColorStop(0, '#101622');
        bladeGrad.addColorStop(0.6, '#00f0ff');
        bladeGrad.addColorStop(1, '#e6ffff');
      }

      ctx.fillStyle = bladeGrad;
      ctx.fill();
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.restore();
    }

    // 4. Center Nose Cone & Shaft Hub
    const hubGrad = ctx.createRadialGradient(0, 0, 5, 0, 0, 36);
    hubGrad.addColorStop(0, '#ffffff');
    hubGrad.addColorStop(0.4, '#1b2333');
    hubGrad.addColorStop(1, '#080a0f');

    ctx.beginPath();
    ctx.arc(0, 0, 36, 0, Math.PI * 2);
    ctx.fillStyle = hubGrad;
    ctx.fill();
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.6)';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.restore();

    // Advance rotation based on current virtual RPM
    // 60 fps * 60 = 3600 frames/min
    const rotationSpeed = (rpm / 3600) * Math.PI * 2;
    this.angle += Math.max(0.02, rotationSpeed);
  }
}

class OscilloscopeVisualizer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.t = 0;
  }

  draw(synthPulse, realPulse, shockPulse) {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    const drawWave = (freq, phase, color, isPulsing) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = isPulsing ? 2.5 : 1.5;
      if (isPulsing) {
        ctx.shadowColor = color;
        ctx.shadowBlur = 10;
      } else {
        ctx.shadowBlur = 0;
      }

      ctx.beginPath();
      for (let x = 0; x < w; x++) {
        const rad = (x * 0.05 * freq) + phase + this.t;
        const y = (h / 2) + Math.sin(rad) * (h * 0.32);
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    };

    // 1. Synthetic Injector Wave (Cyan/Purple)
    drawWave(1.2, 0.5, '#b026ff', synthPulse);
    // 2. Real-World Injector Wave (Green)
    drawWave(0.8, 1.8, '#00ff88', realPulse);
    // 3. Shock Injector Wave (Red)
    drawWave(0.5, 3.14, '#ff1e56', shockPulse);

    this.t += 0.04;
  }
}
