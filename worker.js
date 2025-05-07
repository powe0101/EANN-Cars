importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0");

tf.setBackend('webgl').then(() => {
  console.log('Using WebGL backend');
});

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

  remember(exp) {
    this.memory.push(exp);
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
      const target = reward + (!done ? this.gamma * tf.tidy(() =>
        this.model.predict(tf.tensor2d([nextState])).max(1).dataSync()[0]) : 0);
      const qValues = tf.tidy(() => this.model.predict(tf.tensor2d([state])).dataSync());
      qValues[action] = target;
      states.push(state);
      targets.push(qValues);
    }

    await this.model.fit(tf.tensor2d(states), tf.tensor2d(targets), { epochs: 1, verbose: 0 });
    if (this.epsilon > this.epsilonMin) this.epsilon *= this.epsilonDecay;
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
}

const agent = new DQNAgent(3, 3);

onmessage = async (e) => {
  const { type, data, requestId } = e.data;

  if (type === "experience") {
    agent.remember(data);
  } else if (type === "train") {
    await agent.replay();
  } else if (type === "act") {
    const action = agent.act(data);
    postMessage({ type: "action", requestId, action });
  }
};
