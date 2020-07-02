import React from "react";
import "../Css/selfie.css";
import WebCam from "./selfie";
import { styled } from "@material-ui/core/styles";
import Card from "@material-ui/core/Card";
import CardActions from "@material-ui/core/CardActions";
import CardContent from "@material-ui/core/CardContent";
import Typography from "@material-ui/core/Typography";
import EnhancedEncryptionIcon from "@material-ui/icons/EnhancedEncryption";
import Container from "@material-ui/core/Container";
import Skeleton from "@material-ui/lab/Skeleton";
import CameraFrontIcon from "@material-ui/icons/CameraFront";
import history from "../history";
import verifyIdentity from "../Assets/SelfiePage1.png";
import PlaceholderImage from "../Assets/download.jpg";

const MySkeleton = styled(Skeleton)({
  alignItems: "center",
  textalign: "center",
});
const logOut = () => {
  localStorage.removeItem("auth");
  history.push("/");
};

class TakePhoto extends React.Component {
  constructor(props) {
    super(props);
    this.webcam = null;

    this.state = {
      image: "",
      error: "",
      clickme: false,
      retake: false,
    };
  }
  componentDidMount() {
    this.CanvasElement = document.createElement("canvas");
    this.con = document.getElementById("cancamera");
    console.log(this.con);
    this.context = this.con.getContext("2d");
    console.log(this.context);
    this.video = document.createElement("video");
    this.video.src = "urlToVideo.mp4";
    this.video.width = "300";
    this.video.height = "200";
    this.video.play();
    this.webcam = new WebCam(this.video, this.CanvasElement, this.context);
    this.webcam.SetUp().catch((err) => {
      alert(err);
    });
    this.webcam.setup2();
  }
  handleOnClick = (ev) => {
    const name = ev.target.name;
    if (name === "click") {
      const img = this.webcam.takeBase64Image();
      this.setState({ img: img.base64, clickme: !this.state.clickme });
      console.log(img.base64.length);
      sessionStorage.setItem("img", img.base64);
    } else if (name === "retake") {
      this.setState({ clickme: !this.state.clickme });
    } else if (name === "looksgood") {
      this.setState({ retake: !this.state.retake });
    } else if (name === "submit") {
      if (sessionStorage.getItem("img")) {
        this.webcam.destroy();
        history.push("/docs");
      } else {
        alert("click your selfie for submit");
      }
    }

    ev.preventDefault();
  };

  render() {
    const { img, error, clickme, retake } = this.state;

    const clickbutton = !error ? (
      <button
        id="HandlaSelfieClickButton"
        name="click"
        className="button"
        disabled={clickme}
        onClick={this.handleOnClick}
        style={{ backgroundColor: "green" }}
      >
        CLICK <CameraFrontIcon fontSize="large" />
      </button>
    ) : null;
    const retakebutton = (
      <button
        id="TwoButtons"
        style={{ backgroundColor: "green" }}
        name="retake"
        className="button"
        disabled={!clickme || retake}
        onClick={this.handleOnClick}
      >
        <strong>RETAKE</strong>
      </button>
    );
    // const looksgoodbutton = (
    //   <button
    //     id="TwoButtons"
    //     name="looksgood"
    //     className="button"
    //     disabled={!clickme || retake}
    //     onClick={this.handleOnClick}
    //     style={{ backgroundColor: "green" }}
    //   >
    //     <strong>LOOKSGOOD</strong>
    //   </button>
    // );
    const IMAGE = img ? sessionStorage.getItem("img") : PlaceholderImage;

    return (
      <div className="App-header">
        {localStorage.getItem("auth") ? (
          <button onClick={logOut}>logout</button>
        ) : null}
        <Container id="PageContainer">
          <Card className="HeaderCard" id="HeaderCard">
            <h3 id="SelfiePageDefault" class="text-center default-text">
              Please verify your Identity
            </h3>

            <p id="SelfiePageHeading" className="heading">
              <img src={verifyIdentity} id="VerifyLogo" alt="VerifyLogo"></img>
              Please provide personal details for KYC verification
            </p>
          </Card>

          <Card id="CaptureCard">
            <h3 id="SelfiePageDefault" class="text-center default-text">
              Take a Selfie
            </h3>
            <CardContent color="secondary" id="CaptureCardContent">
              {" "}
              <Typography id="ContentText">
                Make sure your whole face is clearly visible without any glare
                or blur.Make sure that your whole face will fit inside oval
              </Typography>{" "}
            </CardContent>{" "}
            <canvas id="cancamera" width="300" height="200"></canvas>
            <CardActions id="SelfiePageClickButton">{clickbutton}</CardActions>
          </Card>

          <Card id="CaptureCard">
            <h2 id="SelfiePageDefault" class="text-center default-text">
              Preview
            </h2>

            <CardContent id="PreviewImage">
              {IMAGE ? (
                <div>
                  <img
                    id="SelfiePagePreviewImage"
                    src={IMAGE}
                    alt="You see your Selfie here "
                  />
                </div>
              ) : (
                <MySkeleton
                  variant="react"
                  width={200}
                  height={200}
                  animation={false}
                ></MySkeleton>
              )}
            </CardContent>

            <CardActions id="SelfiePageTwoButtons">{retakebutton}</CardActions>

            <CardContent id="ContentText">
              <EnhancedEncryptionIcon /> Your Photo will be used only for KYC
              purpose
            </CardContent>

            <button
              id="SelfiePageSubmitButton"
              fullWidth
              name="submit"
              className="button"
              style={{ backgroundColor: "green" }}
              onClick={this.handleOnClick}
            >
              <strong>SAVE & CONTINUE</strong>
            </button>
          </Card>
        </Container>
      </div>
    );
  }
}
export default TakePhoto;
