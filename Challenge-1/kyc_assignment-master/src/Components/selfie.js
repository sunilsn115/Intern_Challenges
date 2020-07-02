function drawvideo(video, context, width, height) {
  context.drawImage(video, 0, 0, width, height);

  context.save();
  context.scale(0.95, 0.9);
  context.beginPath();
  context.strokeStyle = "blue"; 
  context.lineWidth = 2;
  context.arc(
    width / 2,
    height / 2,
    Math.pow((width * width) / 8 + (height * height) / 8, 1 / 2),
    0,
    Math.PI * 2,
    false
  );
  context.stroke();
  context.closePath();
  context.restore();
  setTimeout(drawvideo, 10, video, context, width, height);
}
class WebCam {
  constructor(CameraElement, CanvasElement, CanvasCameraElement) {
    this.CameraElement = CameraElement;
    this.CanvasElement = CanvasElement;
    this.CanvasCameraElement = CanvasCameraElement;
  }
  SetAspectRatio(width, height) {
    const aspectratio = width / height;

    if (width >= height)
      this.CameraElement.width = aspectratio * this.CameraElement.height;
    else this.CameraElement.height = this.CameraElement.width / aspectratio;
  }
  async SetUp() {
    return new Promise((resolve, reject) => {
      if (navigator.mediaDevices.getUserMedia !== "undefined") {
        navigator.mediaDevices
          .getUserMedia({
            audio: false,
            video: {
              facingMode: "user",
              width: { min: 720, ideal: 1280, max: 1080 },
              height: { min: 400, ideal: 720, max: 720 },
            },
          })
          .then((MediaStream) => {
            if ("srcObject" in this.CameraElement) {
              this.CameraElement.srcObject = MediaStream;
            } else {
              this.CameraElement.src = window.URL.createObjectURL(MediaStream);
            }
            this.CameraElement.addEventListener(
              "loadeddata",
              async () => {
                this.SetAspectRatio(
                  this.CameraElement.width,
                  this.CameraElement.height
                );

                this.CameraElement.play();

                resolve();
              },
              false
            );
          })
          .catch(function (err) {
            
              
                   const error= " You denied Permission . if you are using chrome" +
                  ' At the top right, click More. Settings. At the bottom, click Advanced . Under "Privacy and security," click Content settings.' +
                  " Camera.Turn Ask before accessing on or off";
                
            alert(err+error);
          });
      } else {
        reject();
      }
    });
  }
  drawImage() {
    const imgwidth = this.CameraElement.width;
    const imgheight = this.CameraElement.height;

    const context = this.CanvasElement.getContext("2d");
    this.CanvasElement.width = imgwidth;
    this.CanvasElement.height = imgheight;
    context.drawImage(this.CameraElement, 0, 0, imgwidth, imgheight);
    context.strokeStyle = "blue";
    context.lineWidth = 2; 
    context.strokeRect(0, 0, context.canvas.width, context.canvas.height);

    return { imgwidth, imgheight };
  }

  takeBase64Image() {
    const { imgwidth, imgheight } = this.drawImage();
    const base64 = this.CanvasElement.toDataURL("image/jpeg", 1);
    return { base64, imgwidth, imgheight };
  }
  setup2() {
    this.CameraElement.addEventListener(
      "play",
      (async) => {
        console.log(this.context);
        var self = this;
        drawvideo(
          self.CameraElement,
          self.CanvasCameraElement,
          self.CameraElement.width,
          self.CameraElement.height
        );
      },
      false
    );
  }
  destroy() {
    var s = this;
    s.CameraElement.srcObject.getTracks().forEach((track) => {
      track.stop();
    });
  }
}

export default WebCam;
