let inputLength = 12;

let input;
let chars;

let first = true;

function checkInput() {
    if (/^[a-zA-Z]+$/.test(input.value) || input.value.length == 0) {
        return true;
    } else {
        return false;
    }
}

console.log("Zo zo, een beetje mijn code bekijken, hè?");

setTimeout(() => {
    input = document.getElementById('input');
    input.maxLength = inputLength;

    chars = document.getElementById('chars');

    input.addEventListener("keyup", function (event) {
        if (event.keyCode === 13) {
            event.preventDefault();
            submit();
        }
    });
    first = false;
}, 0)

function validateInput() {
    if (checkInput() == true) {
        input.classList.remove("invalidInput");
    } else {
        input.classList.add("invalidInput");
    }

    let charsLeft = inputLength - input.value.length;
    chars.innerHTML = charsLeft;
    chars.style.marginLeft = (charsLeft >= 10) ? "-48px" : "-32px";
    // chars.innerHTML = input.value.length;
}

function submit() {
    let outputText = document.getElementById("outputText");
    // let inputText = input.value.split(" ");

    // if (!existsInData(inputText[1])) {
    //     let obj = {
    //         "word": inputText[1],
    //         "article": inputText[0]
    //     };
    //     data.words.push(obj);
    //     console.log(obj);
    //     output.innerHTML = `${inputText[1]} is toegevoegd aan de database`
    // } else {
    //     output.innerHTML = `${inputText[1]} bestaat al`;
    // }



    // input.value = "";

    if (checkInput() == true && input.value.length > 0) {
        let output = brain.feed(tokenizeWord(input.value))[0] * 100;
        outputText.innerHTML = (output >= 50) ? `het ${input.value} (${Math.round(output)}%)` : `de ${input.value} (${100 - Math.round(output)}%)`;
        outputText.style.color = "rgb(0, 0, 0)";
    } else if (input.value.length == 0) {
        outputText.innerHTML = "Tja, dan moet je wel wat invullen";
        outputText.style.color = "rgb(5, 128, 255)";
    } else {
        outputText.innerHTML = 'Deze tekens zijn invalide';
        outputText.style.color = "rgb(255, 0, 0)";
    }
}


let data;
function preload() {
    data = loadJSON("./data/data.json");
    brain = new network();
    brain.load("./data/nn.json");
}

let trainBool = false;
function setup() {
    // console.log(data);
    if (trainBool) interval = setInterval(trainInit, 0);
    // brain.learningRate = 0.001;
}

function existsInData(string) {
    for (let i in data.words) {
        if (data.words[i].word == string) {
            return true;
        }
    }
    return false;
}




// AI

function tokenizeWord(word) {
    let array = [];
    let txt = word.toLowerCase().split("");

    for (let i = 0; i < 12; i++) {
        if (txt[txt.length - i - 1]) array.push(txt[txt.length - i - 1].charCodeAt(0) - 96);
        else array.push(0);
    }

    return array;
}

function tokenizeData() {
    for (let i in data.words) {
        data.tokens.push({
            "token": tokenizeWord(data.words[i].word),
            "article": parseInt(data.words[i].article == "de" ? 0 : 1)
        });
    }
}

let brain = new network([12, 10, 10, 1], 0.01);

let avg = 0;
function train() {
    avg = 0;
    for (let i in data.tokens) {
        brain.train(data.tokens[i].token, [data.tokens[i].article]);
        avg += brain.cost / data.tokens.length;
    }
    // console.log(avg);
}

function trainInit() {
    train();
    let assessment;
    if (n % 10 == 0) {
        console.log(avg)
        assessment = assess();
        // brain.learningRate = calcLearningRate(n / 10);
        // console.log("new learning rate: " + brain.learningRate)
        if (assessment == 1) {
            clearInterval(interval);
            brain.save("nn-hundredpercent.json");
        }
    }
    n++;
}

let n = 0;
let interval;
// setInterval(train, 0);

function assess() {
    let percentage = 0;
    for (let i in data.tokens) {
        if (Math.abs(brain.feed(data.tokens[i].token)[0] - data.tokens[i].article) < 0.5) percentage += 1 / data.tokens.length;
        // if (Math.abs(0 - data.tokens[i].article) < 0.5) percentage += 1 / data.tokens.length;
    }
    console.log(`${Math.round(percentage * 1000) / 10}% correct`);
    document.getElementById('outputText').innerHTML = `${Math.round(percentage * 1000) / 10}% correct`;
    return percentage;
}


// function calcLearningRate(epoch) {
//     if (epoch < 5) return 0.005;
//     else return 0.005 * Math.exp(0.0001 * -epoch);
// }