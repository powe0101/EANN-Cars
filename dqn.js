const canvas = document.getElementById('dqnCanvas');
const ctx = canvas.getContext('2d');

const finishLine = { x: 680, y: 480 };

function isOnTrack(x, y) {
  if (x > 100 && x < 700 && y > 100 && y < 500) {
    if (x > 300 && x < 500 && y > 250 && y < 350) return false;
    return true;
  }
  return false;
}

function isOnFinish(x, y) {
  return x > finishLine.x - 20 && x < finishLine.x + 20 &&
         y > finishLine.y - 20 && y < finishLine.y + 20;
}

let requestIdCounter = 0;
const pendingActions = new Map();

const worker = new Worker("worker.js");
worker.onmessage = (e) => {
  const { type, requestId, action } = e.data;
  if (type === "action" && pendingActions.has(requestId)) {
    pendingActions.get(requestId)(action);
    pendingActions.delete(requestId);
  }
};

class Car {
  constructor(x, y, worker) {
    this.x = x;
    this.y = y;
    this.angle = 0;
    this.speed = 2;
    this.size = 20;
    this.sensors = [-Math.PI / 4, 0, Math.PI / 4];
    this.sensorLength = 100;
    this.alive = true;
    this.agentWorker = worker;
    this.lastDistance = this.distanceToFinish();
    this.pastPositions = [];
    this.angleDelta = 0;
  }

  getSensorInputs() {
    return this.sensors.map(offset => {
      const sensorEnd = this.castSensor(offset);
      const dist = sensorEnd?.distance ?? this.sensorLength;
      return 1 - (dist / this.sensorLength);
    });
  }

  castSensor(angleOffset) {
    const angle = this.angle + angleOffset;
    for (let i = 0; i < this.sensorLength; i++) {
      const testX = this.x + Math.cos(angle) * i;
      const testY = this.y + Math.sin(angle) * i;
      if (!isOnTrack(testX, testY)) return { distance: i };
    }
    return { distance: this.sensorLength };
  }

  distanceToFinish() {
    const dx = this.x - finishLine.x;
    const dy = this.y - finishLine.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  update() {
    if (!this.alive) return;

    const state = this.getSensorInputs();
    const requestId = requestIdCounter++;

    pendingActions.set(requestId, (action) => {
      const prevAngle = this.angle;
      if (action === 0) this.angle -= 0.05;
      else if (action === 2) this.angle += 0.05;
      this.angleDelta = this.angle - prevAngle;

      this.x += Math.cos(this.angle) * this.speed;
      this.y += Math.sin(this.angle) * this.speed;

      const reward = this.computeReward();
      const nextState = this.getSensorInputs();
      const done = !isOnTrack(this.x, this.y) || isOnFinish(this.x, this.y);

      this.agentWorker.postMessage({
        type: "experience",
        data: { state, action, reward, nextState, done }
      });

      if (done) this.alive = false;
    });

    this.agentWorker.postMessage({
      type: "act",
      requestId,
      data: state
    });
  }

  computeReward() {
    const nowDist = this.distanceToFinish();
    const delta = this.lastDistance - nowDist;

    this.pastPositions.push({ x: this.x, y: this.y });
    if (this.pastPositions.length > 20) this.pastPositions.shift();
    const dx = this.pastPositions[this.pastPositions.length - 1].x - this.pastPositions[0].x;
    const dy = this.pastPositions[this.pastPositions.length - 1].y - this.pastPositions[0].y;
    const displacement = Math.sqrt(dx * dx + dy * dy);

    this.lastDistance = nowDist;

    if (isOnFinish(this.x, this.y)) return 100;
    if (!isOnTrack(this.x, this.y)) return -100;

    let reward = 0;
    reward += (delta > 0) ? delta * 10 : -1;
    if (displacement < 20) reward -= 2;

    const sensors = this.getSensorInputs();
    for (const s of sensors) {
      if (s > 0.7) reward -= 5;
    }

    if (Math.abs(this.angleDelta) < 0.01) reward += 0.5;
    else reward -= 0.5;

    // NaN 방지
    if (isNaN(reward)) {
      console.warn("NaN 보상 발생", sensors, delta, displacement);
      reward = -10;
    }

    return reward;
  }

  draw(ctx) {
    ctx.save();
    ctx.translate(this.x, this.y);
    ctx.rotate(this.angle);
    ctx.fillStyle = 'blue';
    ctx.fillRect(-this.size / 2, -this.size / 2, this.size, this.size);
    ctx.restore();
  }
}

function drawTrack() {
  ctx.fillStyle = 'lightgreen';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = 'black';
  ctx.fillRect(100, 100, 600, 400);
  ctx.fillStyle = 'lightgreen';
  ctx.fillRect(300, 250, 200, 100);
  ctx.fillStyle = 'yellow';
  ctx.fillRect(finishLine.x - 20, finishLine.y - 20, 40, 40);
}

const carCount = 10;
let cars = [];
let generation = 1;
let stepCounter = 0;

function initializeCars() {
  cars = [];
  for (let i = 0; i < carCount; i++) {
    cars.push(new Car(150, 150 + i * 10, worker));
  }
}

function animate() {
  drawTrack();

  let allDead = true;
  for (const car of cars) {
    if (car.alive) {
      car.update();
      allDead = false;
    }
    car.draw(ctx);
  }

  if (allDead) {
    initializeCars();
    generation += 1;
  }

  stepCounter++;
  if (stepCounter % 10 === 0) {
    worker.postMessage({ type: "train" });
  }

  ctx.fillStyle = 'black';
  ctx.fillText(`Generation: ${generation}`, 10, 20);

  requestAnimationFrame(animate);
}

initializeCars();
animate();
