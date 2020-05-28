// const gpu = new GPU();

const activationFunctions = {
  SIGMOID: 1,
  TANH: 2,
  RELU: 3,
  NONE: 4
};

class neuron {
  constructor(layernr, neuronnr) {
    this.layer = parseInt(layernr);
    this.neuron = parseInt(neuronnr);
    this.weight = [];
    this.bias = [];
    this.activation = 0;
  }

  setWeight(input, input2) {
    if (!input2) this.weight = input;
    else this.weight[input] = input2;
    return this;
  }

  setBias(input) {
    this.bias = input;
    return this;
  }

  addWeight(input) {
    this.weight.push(input);
  }

  log() {
    console.log(this);
  }
}

class network {
  constructor(_structure, _learningRate = 1) {
    this.learningRate = _learningRate;
    if (_structure) {
      if (typeof _structure == "string") {
        this.activationFunction = activationFunctions.SIGMOID;
        this.learningRate = 1;
        this.network = [];
        this.load(_structure, _learningRate);
      } else if (Array.isArray(_structure) && _structure.length >= 2) {
        this.activationFunction = activationFunctions.SIGMOID;

        // Create network and store in 'this.network'
        this.network = [];
        for (let i in _structure) {
          i = parseInt(i);
          let _neuronsInLayer = _structure[i];
          let _layer = [];
          for (let j = 0; j < _neuronsInLayer; j++) {
            let _n = new neuron(i, j);
            for (let k in this.network[i - 1]) {
              _n.addWeight(Math.random() * 2 - 1);
            }
            _n.bias = (Math.random() * 2 - 1);
            _layer.push(_n);
          }
          this.network.push(_layer);
        }
      } else {
        console.error("Constructor argument is not a valid type. Expected: string or 2+ element array");
      }
    } else {
      this.activationFunction = activationFunctions.SIGMOID;
      this.network = [];
    }
  }

  log() {
    this.forEachNeuron((layernr, neuronnr) => {
      this.getNeuron(layernr, neuronnr).log()
    });
  }

  getNeuron(layernr, neuronnr) {
    layernr = parseInt(layernr);
    neuronnr = parseInt(neuronnr);

    if (layernr < 0) layernr = this.network.length + layernr;
    if (layernr >= this.network.length) {
      console.error(`Layer ${layernr} doesn't exist`);
    }
    if (neuronnr >= this.network[layernr].length || neuronnr < 0) {
      console.error("That neuron doesn't exist");
    }
    return this.network[layernr][neuronnr];
  }

  getLayer(layernr) {
    layernr = parseInt(layernr);

    if (layernr > this.network.length - 1) console.error("That layer doesn't exist");
    else if (layernr < -this.network.length) console.error("That layer doesn't exist");

    if (layernr < 0 && layernr >= -this.network.length + 1) layernr = this.network.length + layernr;
    return this.network[layernr];
  }

  forEachLayer(callback, backwards) {
    if (!backwards) {
      for (let i in this.network) {
        callback(parseInt(i));
      }
    } else {
      for (let i in this.network) {
        callback(parseInt(this.network.length - 1 - i));
      }
    }
  }

  forEachNeuron(callback, backwards) {
    if (!backwards) {
      for (let i in this.network) {
        for (let j in this.network[i]) {
          callback(parseInt(i), parseInt(j));
        }
      }
    } else {
      for (let i in this.network) {
        for (let j in this.network[this.network.length - 1 - i]) {
          callback(parseInt(this.network.length - 1 - i), parseInt(j));
        }
      }
    }
  }

  forEachNeuronInLayer(callback, layernr) {
    let layer = this.getLayer(layernr);
    for (let i in layer) {
      if (layer) callback(layernr, i);
    }
  }

  getWeight(n, n2) {
    if (n2.layer - n.layer != 1) console.error("These neurons don't have a connection");
    return this.network[n2.layer][n2.neuron].weight[n.neuron];
  }

  setWeight(n, n2, num) {
    if (n2.layer - n.layer != 1) console.error("These neurons don't have a connection");
    this.network[n2.layer][n2.neuron].weight[n.neuron] = num;
    return this;
  }

  train(input, expectedOutput) {
    let res = this.feed(input);
    this.globalError = 0;
    let cost = 0;

    this.forEachNeuronInLayer((layernr, neuronnr) => {
      let n = this.getNeuron(layernr, neuronnr);
      let grad = this.deriveNormalize(n.activation);
      let error = expectedOutput[neuronnr] - n.activation;
      n.error = grad * error;
      cost += Math.abs(error);
      n.bias += this.learningRate * n.error;
    }, -1);
    this.cost = cost;

    this.forEachNeuron((layernr, neuronnr) => {
      let currentNeuron = this.getNeuron(layernr, neuronnr);
      if (layernr <= this.network.length - 2) {
        let nextLayer = this.getLayer(layernr + 1);

        let sum = 0;
        for (let i in nextLayer) {
          sum += this.getWeight(currentNeuron, nextLayer[i]) * nextLayer[i].error;
        }

        currentNeuron.error = sum * this.deriveNormalize(currentNeuron.activation);
        this.globalError += Math.abs(currentNeuron.error);
        currentNeuron.bias += this.learningRate * currentNeuron.error;

        for (let index in nextLayer) {
          let w = this.getWeight(currentNeuron, nextLayer[index]) + this.learningRate * nextLayer[index].error * currentNeuron.activation;
          this.setWeight(currentNeuron, nextLayer[index], w);
        }
      }
    }, true);

    return res;
  }

  trainGPU(input, expectedOutput) {
    let res = this.feedGPU(input);
    this.globalError = 0;
    let cost = 0;

    this.forEachNeuronInLayer((layernr, neuronnr) => {
      let n = this.getNeuron(layernr, neuronnr);
      let grad = this.deriveNormalize(n.activation);
      let error = expectedOutput[neuronnr] - n.activation;
      n.error = grad * error;
      cost += Math.abs(error);
      n.bias += this.learningRate * n.error;
    }, -1);
    this.cost = cost;

    this.forEachNeuron((layernr, neuronnr) => {
      let currentNeuron = this.getNeuron(layernr, neuronnr);
      if (layernr <= this.network.length - 2) {
        let nextLayer = this.getLayer(layernr + 1);

        let sum = 0;
        for (let i in nextLayer) {
          sum += this.getWeight(currentNeuron, nextLayer[i]) * nextLayer[i].error;
        }

        currentNeuron.error = sum * this.deriveNormalize(currentNeuron.activation);
        this.globalError += Math.abs(currentNeuron.error);
        currentNeuron.bias += this.learningRate * currentNeuron.error;

        for (let index in nextLayer) {
          let w = this.getWeight(currentNeuron, nextLayer[index]) + this.learningRate * nextLayer[index].error * currentNeuron.activation;
          this.setWeight(currentNeuron, nextLayer[index], w);
        }
      }
    }, true);

    return res;
  }

  autoTrain(inputArray, expectedOutputArray) {
    if (expectedOutputArray) {
      for (let i in inputArray) {
        this.train(inputArray[i], expectedOutputArray[i]);
      }
    } else {
      for (let i in inputArray) {
        this.train(inputArray[i][0], inputArray[i][1]);
      }
    }
  }

  feed(input) {
    // Error handling
    let error = false;
    if (!error && !input) {
      console.error("Please provide the input");
      error = true;
    }
    if (!error && input.length != this.network[0].length) {
      console.error(`The input array must be of the same length as the input layer, it is now ${input.length}, but it should be ${this.network[0].length}`);
      error = true;
    }
    if (!error && !Array.isArray(input)) {
      console.error("Please check if the input is an array");
      error = true;
    }
    // if (error) process.exit();

    //Set input layer
    this.forEachNeuronInLayer((i, j) => {
      this.getNeuron(i, j).activation = input[j];
    }, 0);

    this.forEachNeuron((layernr, neuronnr) => {
      if (layernr != 0) {
        let current = this.getNeuron(layernr, neuronnr);
        // let currentLayer = this.getLayer(layernr);
        let prevLayer = this.getLayer(layernr - 1);

        let total = 0;
        for (let l in current.weight) {
          total += current.weight[l] * prevLayer[l].activation;
        }
        total += parseFloat(current.bias);
        current.activation = this.normalize(total);
      }
    });

    // Format output
    let res = [];
    this.forEachNeuronInLayer((layernr, neuronnr) => {
      res.push(parseFloat(this.getLayer(-1)[neuronnr].activation));
    }, -1)
    return res;
  }

  feedGPU(input) {
    let weightMatrix = [];
    let activationMatrix = [];
    let biasMatrix = [];

    let result;

    this.forEachLayer((layernr) => {
      if (layernr == 0) {
        this.forEachNeuronInLayer((layernr) => {
          this.getNeuron(layernr, neuronnr).activation = input[neuronnr];
          activationMatrix.push([input[neuronnr]]);
        });
      } else {
        let currentLayer = this.getLayer(layernr);
        let prevLayer = this.getLayer(layernr - 1);

        weightMatrix = [];
        activationMatrix = [];
        biasMatrix = [];

        for (let i in prevLayer) {
          activationMatrix.push([prevLayer[i].activation]);
        }
        for (let i in currentLayer) {
          weightMatrix.push([]);
          for (let j in currentLayer[i].weight) {
            weightMatrix[i].push(currentLayer[i].weight[j]);
          }
          biasMatrix.push(currentLayer[i].bias);
        }

        let size = this.getLayer(layernr).length;
        // console.log(size, multiplyMatrixOutputSize);
        // if (size != multiplyMatrixOutputSize) {
        //   multiplyMatrix = gpu.createKernel(function (a, b, c) {
        //     let sum = 0;
        //     for (let i = 0; i < c; i++) {
        //       sum += a[this.thread.y][i] * b[i][this.thread.x];
        //     }
        //     return sum;
        //   }).setOutput([1, size]);
        //   multiplyMatrixOutputSize = size;
        // }

        console.log(weightMatrix, activationMatrix, biasMatrix, size);
        if (size != multiplyMatrixOutputSize) multiplyMatrix.setOutput([1, size]);
        multiplyMatrixOutputSize = size;

        result = multiplyMatrix(weightMatrix, activationMatrix, size);

        //currentLayer and result have the same length
        for (let i in currentLayer) {
          result[i] = parseFloat(result[i]);
          result[i] += biasMatrix[i];

          this.getNeuron(layernr, i).activation = this.normalize(result[i]);
        }
      }
    }, false);

    // Format output
    let res = [];
    this.forEachNeuronInLayer((layernr, neuronnr) => {
      res.push(parseFloat(this.getLayer(-1)[neuronnr].activation));
    }, -1);
    return res;
  }

  normalize(input) {
    switch (this.activationFunction) {
      case activationFunctions.SIGMOID:
        return 1 / (1 + Math.pow(Math.E, -input));
        break;
      case activationFunctions.TANH:
        return 2 / (1 + Math.pow(Math.E, -2 * input));
        break;
      case activationFunctions.RELU:
        return Math.max(input, 0);
        break;
      case activationFunctions.NONE:
        return input;
        break;
      default:
        console.error("Invalid activation function");
    }
  }

  deriveNormalize(input) {
    switch (this.activationFunction) {
      case activationFunctions.SIGMOID:
        return input * (1 - input);
        break;
      case activationFunctions.TANH:
        return 1 - Math.pow((2 / (1 + Math.pow(Math.E, -2 * input))), 2);
        break;
      case activationFunctions.RELU:
        return input;
        break;
      case activationFunctions.NONE:
        return input;
        break;
      default:
        console.error("Invalid activation function");
    }
  }

  async save(path, options) {
    // Save the network to a file
    let obj = [];
    obj.push({
      learningRate: this.learningRate,
      activationFunction: this.activationFunction
    });

    this.forEachLayer((layernr) => {
      let layer = [];
      this.forEachNeuronInLayer((layernr, neuronnr) => {
        let current = this.getNeuron(layernr, neuronnr);
        layer.push({
          bias: current.bias,
          weight: current.weight,
        });
      }, layernr);
      obj.push(layer);
    });

    // if (options && options.forceOverwrite && options.forceOverwrite == true) {
    // fs.writeFile(path, JSON.stringify(obj), (error) => {
    //   if (error) throw error;
    // });
    saveJSON(obj, path);
    // } else {
    // fs.writeFile(path, JSON.stringify(obj), {
    // flag: 'wx'
    // }, (error) => {
    // if (error && error.code == "EEXIST") {
    // console.warn("[MARVIN.JS WARNING] That file already exists. Try the forceOverwrite flag if you'd like to overwrite");
    // } else if (error) {
    // throw error;
    // }
    // });
    // }
  }

  async load(path, callback) {
    let response = await fetch(path);
    let data = await response.json();

    this.learningRate = data[0].learningRate;
    this.activationFunction = data[0].activationFunction;

    data.shift();

    this.network = [];
    for (let i in data) {
      i = parseInt(i);
      let _layer = [];
      for (let j in data[i]) {
        let _n = new neuron(i, j);
        _n.weight = data[i][j].weight;
        _n.bias = data[i][j].bias;
        _layer.push(_n);
      }
      this.network.push(_layer);
    }

    if (callback) callback();
    return;
  }

  getParamCount() {
    let params = 0;
    this.forEachLayer((i) => {
      params += this.network[i].length;
      if (i != 0) params += this.network[i].length * this.network[i - 1].length;
    });
    return params;
  }
}

class population {
  constructor(structure, size, mutationRate, percentile) {
    this.population = [];
    this.generation = 0;
    this.network = 0;
    this.structure = structure;
    this.maxMutation = 1;

    let generation = [];
    for (let i = 0; i < size; i++) {
      let n = new network(structure, 0);
      generation.push({
        network: n,
        id: i,
        fitness: null
      });
    }
    this.size = size;
    this.population = generation;
    this.mutationRate = mutationRate;
    this.percentile = percentile;
  }

  getCurrentGeneration() {
    return this.population;
  }

  getCurrentNetwork() {
    return this.population[this.network].network;
  }

  cycle() {
    this.network++;
    if (this.network >= this.size) {
      this.nextGeneration();
      this.network = 0;
    }
  }

  selectBestNetwork() {
    let sorted = this.population.sort(function (a, b) {
      return b.fitness - a.fitness;
    });
    console.log(sorted[0]);
    this.network = sorted[0].id;
  }

  nextGeneration() {
    let sorted = this.population.sort(function (a, b) {
      return b.fitness - a.fitness;
    });
    let highestIndex = Math.floor(sorted.length * this.percentile);
    sorted.splice(highestIndex, sorted.length - highestIndex + 1)
    console.log(sorted[0].fitness);

    let prevGeneration = this.population;
    let generation = [];
    this.population = [];
    for (let i = 0; i < this.size; i++) {
      let j = floor(i / (this.size / sorted.length));
      // let net = sorted[j].network;

      let net = new network(this.structure, 0);
      let tempNet = [];
      let layer = [];
      sorted[j].network.forEachNeuron((layernr, neuronnr) => {
        let parentNet = sorted[j].network;
        let n = new neuron(layernr, neuronnr);
        n.activation = parentNet.network[layernr][neuronnr].activation;

        if (Math.random() <= this.mutationRate && this.network != 0) {
          n.bias = Math.random() * this.maxMutation * 2 - this.maxMutation;
        } else {
          n.bias = parentNet.network[layernr][neuronnr].bias;
        }
        for (let k in parentNet.network[layernr][neuronnr].weight) {
          if (Math.random() <= this.mutationRate && this.network != 0) {
            n.weight.push(Math.random() * this.maxMutation * 2 - this.maxMutation);
          } else {
            n.weight.push(parentNet.network[layernr][neuronnr].weight[k]);
            // n.weight.push(i);
          }
        }
        layer.push(n);
        if (neuronnr == this.structure[layernr] - 1) {
          tempNet.push(layer);
          layer = [];
        }
      });
      net.network = [...tempNet];

      // net.forEachNeuron((layernr, neuronnr) => {
      //   let neuron = net.getNeuron(layernr, neuronnr);
      //   // neuron.bias = i;
      //   for (let k in neuron.weight) {
      //     neuron.weight[k] = i/20;
      //   }
      //   net.network[layernr][neuronnr] = neuron;
      // });
      this.population.push({
        network: net,
        id: i,
        fitness: null
      })
      // console.log(this.population);
      // console.log(this.population[i].network.network[1][0].weight[0], this.population[i].network.network[1][0]);
    }
    // this.population = generation;
    this.generation++;
  }
}


// let multiplyMatrix = gpu.createKernel(function (a, b, c) {
//   let sum = 0;
//   for (let i = 0; i < c; i++) {
//     sum += a[this.thread.y][i] * b[i][this.thread.x];
//   }
//   return sum;
// }).setOutput([1, 2]);
// multiplyMatrix.dynamicOutput = true;
// let multiplyMatrixOutputSize;

// let brain = new network([3, 4, 4, 3]);
// console.log(brain.train([0.5, 0.3, 0.4], [0.5, 0.3, 0.4]));
// console.log(brain.trainGPU([0.5, 0.3, 0.4], [0.5, 0.3, 0.4]));


// let timeA = new Date();
// for (let i = 0; i < 100; i++) {
//   brain.train([0.5, 0.3, 0.4], [0.5, 0.3, 0.4]);
// }
// let timeB = new Date();
// let diff = timeB - timeA;
// console.log(`CPU: ${diff} ms`)

// timeA = new Date();
// for (let i = 0; i < 100; i++) {
//   brain.trainGPU([0.5, 0.3, 0.4], [0.5, 0.3, 0.4]);
// }
// timeB = new Date();
// diff = timeB - timeA;
// console.log(`GPU: ${diff} ms`)





// let brain = new network([2, 2]);
// brain.activationFunction = activationFunctions.NONE;
// console.log(brain.feed([0.5, 0.3]));
// console.log(brain.feedGPU([0.5, 0.3]));
