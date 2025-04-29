// 캔버스 초기화
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
canvas.width = 800;
canvas.height = 600;
let trackImage; // 트랙을 미리 그려놓을 임시 캔버스

const finishLine = { x: 650, y: 450, width: 50, height: 50 }; // 도착지점

// 신경망
class NeuralNetwork {
  constructor(inputCount, hiddenCount, outputCount) {
    this.weightsInputHidden = new Array(hiddenCount).fill(0).map(() =>
      new Array(inputCount).fill(0).map(() => Math.random() * 2 - 1)
    );
    this.weightsHiddenOutput = new Array(outputCount).fill(0).map(() =>
      new Array(hiddenCount).fill(0).map(() => Math.random() * 2 - 1)
    );
  }

  activate(x) {
    return 1 / (1 + Math.exp(-x));
  }

  predict(inputs) {
    const hidden = this.weightsInputHidden.map(weights => 
      this.activate(weights.reduce((sum, w, i) => sum + w * inputs[i], 0))
    );
    const output = this.weightsHiddenOutput.map(weights => 
      weights.reduce((sum, w, i) => sum + w * hidden[i], 0)
    );
    return output;
  }

  clone() {
    const clone = new NeuralNetwork(0,0,0);
    clone.weightsInputHidden = this.weightsInputHidden.map(row => [...row]);
    clone.weightsHiddenOutput = this.weightsHiddenOutput.map(row => [...row]);
    return clone;
  }

  mutate(rate = 0.3) { // 변이 확률 30%
    this.weightsInputHidden.forEach(row => 
      row.forEach((w, i) => {
        if (Math.random() < rate) row[i] += (Math.random() * 2 - 1) * 0.5;
      })
    );
    this.weightsHiddenOutput.forEach(row => 
      row.forEach((w, i) => {
        if (Math.random() < rate) row[i] += (Math.random() * 2 - 1) * 0.5;
      })
    );
  }
}

// 차량
class Car {
  constructor(x, y, brain = null) {
    this.x = x;
    this.y = y;
    this.angle = 0;
    this.speed = 2;
    this.size = 20;
    this.sensors = [];
    this.sensorCount = 5;
    this.sensorLength = 100;
    this.sensorSpread = Math.PI / 2;
    this.initSensors();
    this.alive = true;
    this.score = 0;
    this.timeAlive = 0;
    this.maxTime = 500;
    this.startDistanceToFinish = this.distanceToFinish(); // 도착점과 초기 거리
    this.brain = brain ? brain.clone() : new NeuralNetwork(this.sensorCount, 6, 1);
    if (brain) this.brain.mutate(0.3); // 30% 변이
  }

  initSensors() {
    this.sensors = [];
    for (let i = 0; i < this.sensorCount; i++) {
      const sensorAngle = -this.sensorSpread/2 + (i/(this.sensorCount-1)) * this.sensorSpread;
      this.sensors.push(sensorAngle);
    }
  }
  
  update() {
    if (!this.alive) return;
  
    const sensorReadings = this.sensors.map(angleOffset => {
      const sensorEnd = this.castSensor(angleOffset);
      return 1 - (sensorEnd.distance / this.sensorLength);
    });
  
    const output = this.brain.predict(sensorReadings);
    const steering = (output[0] - 0.5) * 2;
  
    this.angle += steering * 0.05;
    this.x += Math.cos(this.angle) * this.speed;
    this.y += Math.sin(this.angle) * this.speed;
  
    const newDistance = this.distanceToFinish();
    const progress = this.startDistanceToFinish - newDistance;
    this.score += progress * 10;
    this.startDistanceToFinish = newDistance;
  
    if (isOnFinish(this.x, this.y)) {
      this.alive = false;
      const bonus = 500 + (this.maxTime - this.timeAlive) * 2; // 도착 시간에 따라 추가 점수
      this.score += bonus;
    }
    else if (this.isColliding()) {
      this.alive = false;
    }
  
    this.timeAlive++;
    if (this.timeAlive > this.maxTime) {
      this.alive = false;
    }
  }
  

  isColliding() {
    return !isOnTrack(this.x, this.y);
  }

  distanceToFinish() {
    const dx = this.x - (finishLine.x + finishLine.width/2);
    const dy = this.y - (finishLine.y + finishLine.height/2);
    return Math.sqrt(dx*dx + dy*dy);
  }

  castSensor(angleOffset) {
    const startX = this.x;
    const startY = this.y;
    const angle = this.angle + angleOffset;
    const dx = Math.cos(angle);
    const dy = Math.sin(angle);

    for (let i = 0; i < this.sensorLength; i++) {
      const testX = startX + dx * i;
      const testY = startY + dy * i;

      if (!isOnTrack(testX, testY)) {
        return { x: testX, y: testY, distance: i };
      }
    }
    return { x: startX + dx * this.sensorLength, y: startY + dy * this.sensorLength, distance: this.sensorLength };
  }

  drawSensors(ctx) {
    for (const angleOffset of this.sensors) {
      const sensorEnd = this.castSensor(angleOffset);
      ctx.beginPath();
      ctx.moveTo(this.x, this.y);
      ctx.lineTo(sensorEnd.x, sensorEnd.y);
      ctx.strokeStyle = 'red';
      ctx.stroke();
    }
  }

  draw(ctx) {
    if (!this.alive) return;
    ctx.save();
    ctx.translate(this.x, this.y);
    ctx.rotate(this.angle);
    ctx.fillStyle = 'blue';
    ctx.fillRect(-this.size/2, -this.size/2, this.size, this.size);
    ctx.restore();
    this.drawSensors(ctx);
  }
}

// 트랙 그리기
function drawTrack(ctx) {
  ctx.fillStyle = 'green';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.fillStyle = 'black';
  ctx.beginPath();
  ctx.moveTo(100, 100);
  ctx.lineTo(700, 100);
  ctx.lineTo(700, 500);
  ctx.lineTo(100, 500);
  ctx.closePath();
  ctx.fill();

  ctx.beginPath();
  ctx.moveTo(200, 200);
  ctx.lineTo(600, 200);
  ctx.lineTo(600, 400);
  ctx.lineTo(200, 400);
  ctx.closePath();
  ctx.fillStyle = 'green';
  ctx.fill();
}

// 트랙 픽셀 체크
function isOnTrack(x, y) {
  const trackCtx = trackImage.getContext('2d');
  const pixel = trackCtx.getImageData(x, y, 1, 1).data;
  return pixel[0] < 50 && pixel[1] < 50 && pixel[2] < 50;
}

// Finish Line 체크
function isOnFinish(x, y) {
  return (x > finishLine.x && x < finishLine.x + finishLine.width &&
          y > finishLine.y && y < finishLine.y + finishLine.height);
}

// 트랙 이미지 생성
function createTrack() {
  const trackCanvas = document.createElement('canvas');
  trackCanvas.width = canvas.width;
  trackCanvas.height = canvas.height;
  const trackCtx = trackCanvas.getContext('2d');
  drawTrack(trackCtx);
  trackImage = trackCanvas;
}

// 차량 관리
let cars = [];
const carCount = 50;
let generation = 1;

function generateCars(baseBrains = []) {
  const cars = [];
  for (let i = 0; i < carCount; i++) {
    const baseBrain = baseBrains.length > 0 ? baseBrains[Math.floor(Math.random() * baseBrains.length)] : null;
    const car = new Car(150 + Math.random() * 20 - 10, 150 + Math.random() * 20 - 10, baseBrain);
    car.angle = (Math.random() * 0.2 - 0.1);
    cars.push(car);
  }
  return cars;
}

cars = generateCars();

// 메인 루프
function animate() {
  ctx.drawImage(trackImage, 0, 0);

  ctx.fillStyle = 'yellow';
  ctx.fillRect(finishLine.x, finishLine.y, finishLine.width, finishLine.height);

  cars.forEach(car => {
    car.update();
    car.draw(ctx);
  });

  if (cars.every(car => !car.alive)) {
    nextGeneration();
  }

  ctx.fillStyle = 'black';
  ctx.fillText(`Generation: ${generation}`, 10, 20);

  requestAnimationFrame(animate);
}

// 다음 세대
function nextGeneration() {
  generation++;

  cars.sort((a, b) => b.score - a.score);

  const survivors = cars.slice(0, Math.floor(carCount * 0.2)); // 상위 20%
  const baseBrains = survivors.map(car => car.brain);

  cars = generateCars(baseBrains);
}

createTrack();
animate();
