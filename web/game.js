const canvas = document.getElementById("game");

const runner = PongWorldRunner.create({ canvas });

function resize() {
  runner.resize(window.innerWidth, window.innerHeight);
}

window.addEventListener("resize", resize);
runner.attachKeyboard();
runner.begin();
resize();

window.__pongClient = runner;
