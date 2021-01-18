

function startBlocks(canvasElement, I, J, s) {
    //var I = 3;
    //var J = 4;
    //var s = 80;
    var n = 0;
    var pose = [
        {x: 0, y: 0},
        {x: 1, y: 1},
        {x: 2, y: 2},
        {x: 0, y: 0},
    ];
    var context = canvasElement.getContext("2d");
    function grid() {
      context.fillStyle = '#000000';
      context.lineWidth = 1;
      context.strokeStyle = '#CCCCC1';
      for (i = 0; i < I + 1; i++) {
        context.beginPath();
        context.moveTo(i * s, 0);
        context.lineTo(i * s, J * s);
        context.stroke();
      }
      for (j = 0; j < J + 1; j++) {
        context.beginPath();
        context.moveTo(0, j * s);
        context.lineTo(I * s, j * s);
        context.stroke();
      }
    }

    var xi = 0;
    var xj = J - 1;
    var di = 1;
    var dj = -1;
    function draw() {
      context.clearRect(0, 0, I * s, J * s);
      grid();
      context.beginPath()
      context.fillStyle = '#000000';
      context.fillRect(xi * s, xj * s, s, s); 

      xi = xi + di;
      xj = xj + dj;
      if (xj === -1 && xi === I) {
        triangle(canvasElement,(I-1)*s,(0)*s,'r',s);
        xi = 0;
        xj = J - 1;
        di = 1;
        dj = -1;
      }
      if (xj === J && xi === I ) {
        // (xj === 0 && xi === 0)|| ) {
        triangle(canvasElement,(I-1)*s,(0)*s,'b',s);
        xi = 0;
        xj = J - 1;
        di = 1;
        dj = -1;
      }
      if (xi >= I) {
        xi = I - 2;
        di = di * -1;
      }
      if (xj >= J) {
        xj = J - 2;
        dj = dj * -1;
      }
      if (xi < 0) {
        xi = 1;
        di = di * -1;
      }
      if (xj < 0) {
        xj = 1;
        dj = dj * -1;
      }
      n = n + 1;
    }
    //window.requestAnimationFrame(draw);
    window.setInterval(draw, 500);
}
