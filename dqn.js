//<!-- TensorFlow.js 추가 -->
//<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0"></script>
//<canvas id="dqnCanvas"></canvas>
//<script>

// === DQN Agent ===
class DQNAgent {
  constructor(stateSize, actionSize) {
    this.stateSize = stateSize;
    this.actionSize = actionSize;
    this.gamma = 0.95;
    this.epsilon = 1.0;
    this.epsilonMin = 0.05;
    this.epsilonDecay = 0.995;
    this.learningRate = 0.001;
    this.batchSize = 32;
    this.memory = [];
    this.model = this.buildModel();
  }

  buildModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [this.stateSize], units: 24, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 24, activation: 'relu' }));
    model.add(tf.layers.dense({ units: this.actionSize }));
    model.compile({ optimizer: tf.train.adam(this.learningRate), loss: 'meanSquaredError' });
    return model;
  }

  act(state) {
    if (Math.random() < this.epsilon) {
      return Math.floor(Math.random() * this.actionSize);
    }
    return tf.tidy(() => {
      const input = tf.tensor2d([state]);
      const qValues = this.model.predict(input);
      return qValues.argMax(1).dataSync()[0];
    });
  }

  remember(state, action, reward, nextState, done) {
    this.memory.push({ state, action, reward, nextState, done });
    if (this.memory.length > 10000) this.memory.shift();
  }

  async replay() {
    if (this.memory.length < this.batchSize) return;
    const minibatch = [];
    while (minibatch.length < this.batchSize) {
      const i = Math.floor(Math.random() * this.memory.length);
      minibatch.push(this.memory[i]);
    }

    const states = [];
    const targets = [];

    for (const { state, action, reward, nextState, done } of minibatch) {
      const target = reward + (!done ? this.gamma * tf.tidy(() => this.model.predict(tf.tensor2d([nextState])).max(1).dataSync()[0]) : 0);
      const qValues = tf.tidy(() => this.model.predict(tf.tensor2d([state])).dataSync());
      qValues[action] = target;
      states.push(state);
      targets.push(qValues);
    }

    await this.model.fit(tf.tensor2d(states), tf.tensor2d(targets), { epochs: 1, verbose: 0 });
    if (this.epsilon > this.epsilonMin) this.epsilon *= this.epsilonDecay;
  }
}

// === 차량 클래스 (센서 및 DQN 통합) ===
class Car {
  constructor(x, y, agent) {
    this.x = x;
    this.y = y;
    this.angle = 0;
    this.speed = 2;
    this.size = 20;
    this.sensors = [-Math.PI/4, 0, Math.PI/4];
    this.sensorLength = 100;
    this.alive = true;
    this.agent = agent;
    this.lastDistance = this.distanceToFinish();
    this.pastPositions = [];
  }

  getSensorInputs() {
    return this.sensors.map(offset => {
      const sensorEnd = this.castSensor(offset);
      return 1 - (sensorEnd.distance / this.sensorLength);
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
    const action = this.agent.act(state);
    if (action === 0) this.angle -= 0.05;
    else if (action === 2) this.angle += 0.05;
    this.x += Math.cos(this.angle) * this.speed;
    this.y += Math.sin(this.angle) * this.speed;

    const reward = this.computeReward();
    const nextState = this.getSensorInputs();
    const done = !isOnTrack(this.x, this.y) || isOnFinish(this.x, this.y);
    this.agent.remember(state, action, reward, nextState, done);
    if (done) this.alive = false;
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
    if (displacement < 20) return -2;
    if (delta < 0) return -1;

    return delta * 10;
  }

  draw(ctx) {
    ctx.save();
    ctx.translate(this.x, this.y);
    ctx.rotate(this.angle);
    ctx.fillStyle = 'blue';
    ctx.fillRect(-this.size/2, -this.size/2, this.size, this.size);
    ctx.restore();
  }
}

// === 환경 ===
const canvas = document.getElementById('dqnCanvas');
const ctx = canvas.getContext('2d');
canvas.width = 800;
canvas.height = 600;

const finishLine = { x: 680, y: 480 };

function isOnTrack(x, y) {
  if (x > 100 && x < 700 && y > 100 && y < 500) {
    if (x > 300 && x < 500 && y > 250 && y < 350) {
      return false;
    }
    return true;
  }
  return false;
}

function isOnFinish(x, y) {
  return x > finishLine.x - 20 && x < finishLine.x + 20 && y > finishLine.y - 20 && y < finishLine.y + 20;
}

// === 다중 차량 초기화 ===
const carCount = 10;
let cars = [];
let generation = 1;

function initializeCars() {
  cars = [];
  for (let i = 0; i < carCount; i++) {
    const agent = new DQNAgent(3, 3);
    cars.push(new Car(150, 150 + i * 10, agent));
  }
}

// === 트랙 ===
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

// === 메인 루프 ===
async function animate() {
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

  for (const car of cars) {
    await car.agent.replay();
  }

  ctx.fillStyle = 'black';
  ctx.fillText(`Generation: ${generation}`, 10, 20);

  requestAnimationFrame(animate);
}

initializeCars();
animate();
</script>
