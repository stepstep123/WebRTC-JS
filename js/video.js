async function main() {
    const video = document.querySelector('video');
    const canvas = document.querySelector('canvas');
    const select = document.querySelector('select');

    const img2 = document.createElement('img');
    // img2.src = '../bg.jpg'
    img2.src = encodeURI('http://r3dg6y3l0.hd-bkt.clouddn.com/WebRTC/background/bg.jpg');

    video.width = 640;
    video.height = 480;
    const webcam = await tf.data.webcam(video);
    const modelUrl = 'http://r3dg6y3l0.hd-bkt.clouddn.com/WebRTC/model/model.json';
    const model = await tf.loadGraphModel(modelUrl);

    // Set initial recurrent state
    let [r1i, r2i, r3i, r4i] = [tf.tensor(0.), tf.tensor(0.), tf.tensor(0.), tf.tensor(0.)];

    // Set downsample ratio
    const downsample_ratio = tf.tensor(0.5);

    // Inference loop
    while(true) {
        await tf.nextFrame();
        const img = await webcam.capture();
        const src = tf.tidy(() => img.expandDims(0).div(255)); // normalize input
        const [fgr, pha, r1o, r2o, r3o, r4o] = await model.executeAsync(
            {src, r1i, r2i, r3i, r4i, downsample_ratio}, // provide inputs
            ['fgr', 'pha', 'r1o', 'r2o', 'r3o', 'r4o']   // select outputs
        );


        // Draw the result based on selected view
        const viewOption = select.value;
        if (viewOption === 'recurrent1') {
            drawHidden(r1o, canvas);
        } else if (viewOption === 'recurrent2') {
            drawHidden(r2o, canvas);
        } else if (viewOption === 'recurrent3') {
            drawHidden(r3o, canvas);
        } else if (viewOption === 'recurrent4') {
            drawHidden(r4o, canvas);
        } else if (viewOption === 'xingkong') {
            console.log("white")
            drawMatte(fgr.clone(), pha.clone(), canvas);
            //canvas.style.background = 'rgb(255, 255, 255)';
            //canvas.style.background = 'url(bg.jpg)';
            document.getElementById('bg').style.background = 'url("http://r3dg6y3l0.hd-bkt.clouddn.com/WebRTC/background/bg.jpg")';
            //canvas.style.background = 'url("http://r3dg6y3l0.hd-bkt.clouddn.com/WebRTC/bg.jpg")';
            //console.log("bg: ",canvas.style.background);

        } else if (viewOption === 'green') {
            console.log("green");
            drawMatte(fgr.clone(), pha.clone(), canvas);
            document.getElementById('bg').style.background = 'rgb(120, 255, 155)';
            // canvas.style.background  = 'rgb(120, 255, 155)';
        } else if (viewOption === 'alpha') {
            drawMatte(null, pha.clone(), canvas);
            canvas.style.background = 'rgb(0, 0, 0)';
        } else if (viewOption === 'foreground') {
            drawMatte(fgr.clone(), null, canvas);
        }

        // Dispose old tensors.
        tf.dispose([img, src, fgr, pha, r1i, r2i, r3i, r4i]);

        // Update recurrent states.
        [r1i, r2i, r3i, r4i] = [r1o, r2o, r3o, r4o];
    }
}

async function drawMatte(fgr, pha, canvas){
    const rgba = tf.tidy(() => {
        const rgb = (fgr !== null) ?
            fgr.squeeze(0).mul(255).cast('int32') :
            tf.fill([pha.shape[1], pha.shape[2], 3], 255, 'int32');
        const a = (pha !== null) ?
            pha.squeeze(0).mul(255).cast('int32') :
            tf.fill([fgr.shape[1], fgr.shape[2], 1], 255, 'int32');
        return tf.concat([rgb, a], -1);
    });
    fgr && fgr.dispose();
    pha && pha.dispose();
    const [height, width] = rgba.shape.slice(0, 2);
    const pixelData = new Uint8ClampedArray(await rgba.data());
    const imageData = new ImageData(pixelData, width, height);
    canvas.width = width;
    canvas.height = height;
    canvas.getContext('2d').putImageData(imageData, 0, 0);
    rgba.dispose();
}

async function drawHidden(r, canvas) {
    const rgba = tf.tidy(() => {
        r = r.unstack(-1)
        r = tf.concat(r, 1)
        r = r.split(4, 1)
        r = tf.concat(r, 2)
        r = r.squeeze(0)
        r = r.expandDims(-1)
        r = r.add(1).mul(127.5).cast('int32')
        r = r.tile([1, 1, 3])
        r = tf.concat([r, tf.fill([r.shape[0], r.shape[1], 1], 255, 'int32')], -1)
        return r;
    });
    const [height, width] = rgba.shape.slice(0, 2);
    const pixelData = new Uint8ClampedArray(await rgba.data());
    const imageData = new ImageData(pixelData, width, height);
    canvas.width = width;
    canvas.height = height;
    canvas.getContext('2d').putImageData(imageData, 0, 0);
    rgba.dispose();
}

window.addEventListener('load', main);
