
            //var canvasElement = document.querySelector("#myCanvas");

            function triangle(canvasElement, x, y, d, s=20, b=0) {
            var context = canvasElement.getContext("2d");
            // the triangle
            if (d == 'l') {
            context.beginPath();
            context.moveTo(x, y);
            context.lineTo(x, y + s);
//              context.lineTo(x + s, y + s);
            context.lineTo(x + s, y);
            context.closePath();
                col = '#448234';
            }
            if (d == 'r') {
            context.beginPath();
            context.moveTo(x, y);
            context.lineTo(x + s, y);
            context.lineTo(x + s, y + s);
//              context.lineTo(x, y + s);
            context.closePath();
                col = '#243294';
            }
            if (d == 'b') {
            context.beginPath();
            context.moveTo(x + s, y);
            context.lineTo(x + s, y + s);
            context.lineTo(x, y + s);
//              context.lineTo(x, y);
            context.closePath();
                col = '#14C284';
            }

            // the outline
            context.lineWidth = 0;
            context.strokeStyle = '#CCCCC1';
            context.stroke();

            // the fill color
            context.fillStyle = col;
            //bb = Math.min(255, b*2.5);
            //context.fillStyle = "rgb(" + bb * lf + ", " + bb * rf + "," + bb * bf + ")";
            //if (b < 6) {
            //    context.fillStyle = "rgb(" + 250 + ", " + 8 + "," + 4 + ")";
            //}
            context.fill();
            }

        function wait(ms){
            var start = new Date().getTime();
            var end = start;
            while(end < start + ms) {
                end = new Date().getTime();
            }
        }

         //triangle(23, 24, 'l');
         //triangle(43, 84, 'r');
         //triangle(83, 84, 'b');
            console.log('func solves');
         function solveS(canvasElement, i, j, draw=false, I=0, J=0, s=20) {
            //console.log('solves');

            if (draw) {
            var context = canvasElement.getContext("2d");
             context.beginPath();
             context.moveTo(0, J * s - 0);
             context.lineTo(i * s, J * s - 0);
             context.lineTo(i * s, J * s - j * s);
             context.lineTo(0, J * s - j * s);
             context.closePath();
             context.lineWidth = 2;
             context.strokeStyle = '#CCCCC1';
             context.stroke();
            }

            //console.log('solve');
             c = 0;
             x = 0;
             y = 0;
             dirx = 1;
             diry = 1;
             d = '';
             done = false;
             b = 0;
            if (draw){
             context.beginPath();
             context.moveTo(s/2, J * s - s/2);
            }
             while (done === false) {
                 //wait(1);
                 x += dirx;
                 y += diry;
            if (draw) {
                context.lineTo(x * s + s/2, J * s - y * s + s/2);
            }
            //console.log('c+=1');
                 c += 1;
                 if (c > 2000) {
                     console.log('long', i, j);
                     done = true;
                 }
                 if (c > 10000) {
                     done = true;
                 }
                 if (x === i - 1 && y === j - 1) {
                     d = 'r';
                     done = true;
                 }
                 if (x === 0 && y === j - 1) {
                     d = 'l';
                     done = true;
                 }
                 if (x === i - 1 && y === 0) {
                     d = 'b';
                     done = true;
                 }
                 if (x === i - 1) {
                     dirx = dirx * -1;
                     b += 1;
                 }
                 if (y === j - 1) {
                     diry = diry * -1;
                     b += 1;
                 }
                 if (x <= 0) {
                     dirx = dirx * -1;
                     b += 1;
                 }
                 if (y <= 0) {
                     diry = diry * -1;
                     b += 1;
                 }
                 //console.log(c, x, y, dirx, diry, d, done, i, j);
             }
                if (draw) {
             context.lineWidth = 1;
             context.strokeStyle = '#CCCCC1';
             context.stroke();
                    console.log(d, b);
                }
                //console.log('d,b',d,b);
             return [d, b];
         }

         //console.log('solve 5,3', solve(5, 3));
         //    solve(2,2,true);
         //    solve(3,3,true);
         //    solve(4,4,true);
         //    solve(5,5,true);
         //    solve(6,6,true);
         //    solve(7,7,true);
         //    solve(8,8,true);
         //    solve(9,9,true);

         console.log('loading');
         function solveR(canvasElement, I, J, s, skip=false) {
            console.log('solvR');
             if (skip) { factor = 2 } else {factor = 1};
            for (i = 2; i < I; i++) {
            //console.log('i');
            //console.log(i);
            //console.log(canvasElement);
                for (j = 2; j < J; j++) {
                    //d, b = solve(i * 2 + 1, j * 2 + 1);
                    //console.log(canvasElement,  i * factor + 0, j * factor + 0, solveS);
                    [d, b] = solveS(canvasElement, i * factor + 0, j * factor + 0);
                    triangle(canvasElement, i * s , J * s  - j * s, d, s, b);
                }
            }
         }
         console.log('done loading');