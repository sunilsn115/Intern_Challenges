import React from "react";
import { withRouter, Redirect } from "react-router-dom";
import "../Css/Docs.css";
import Button from "@material-ui/core/Button";
import history from "../history";
var CryptoJS = require("crypto-js");

class Preview extends React.Component { 
  constructor(props) {
    super(props);
    this.state = {
      name: sessionStorage.getItem("Name")
        ? JSON.parse(
            CryptoJS.AES.decrypt(
              sessionStorage.getItem("Name"),
              "my-secret-key@123"
            ).toString(CryptoJS.enc.Utf8)
          )
        : "",
      dateofbirth: sessionStorage.getItem("DOB")
        ? JSON.parse(
            CryptoJS.AES.decrypt(
              sessionStorage.getItem("DOB"),
              "my-secret-key@123"
            ).toString(CryptoJS.enc.Utf8)
          )
        : "",
      gender: sessionStorage.getItem("gender")
        ? JSON.parse(
            CryptoJS.AES.decrypt(
              sessionStorage.getItem("gender"),
              "my-secret-key@123"
            ).toString(CryptoJS.enc.Utf8)
          )
        : "",
      img: sessionStorage.getItem("img") ? sessionStorage.getItem("img") : "",
      imgfront: sessionStorage.getItem("front")
        ? sessionStorage.getItem("front")
        : "",
      imgback: sessionStorage.getItem("back")
        ? sessionStorage.getItem("back")
        : "",
      id: sessionStorage.getItem("id") ? sessionStorage.getItem("id") : "",
    };
  }
  handleonsubmit = (ev) => {
    const type = ev.target.name;
    if (type === "submit") {
      alert("FORM IS SUBMITTED ");
      localStorage.removeItem("auth");
      localStorage.removeItem("DocsBox");
      sessionStorage.clear();
      history.push("/");
        window.onbeforeunload = function() {
          localStorage.clear();
       }
      window.location.reload();
    }
    ev.preventDefault();
  };
  render() {
    const {
      name,
      dateofbirth,
      gender,
      img,
      imgfront,
      imgback,
      id,
    } = this.state;
    if (!name || !dateofbirth || !gender) {
      alert(
        "You haven't filled Personal Details ,Please fill it by clicking Ok"
      );
      return <Redirect to="/detail" />;
    } else if (!img) {
      alert(
        "You haven't taken selfie ,Please take it before submiting the form by clicking Ok"
      );
      return <Redirect to="/selfie" />;
    } else if (!id || !imgfront || !imgback) {
      alert(
        "You haven't choosen any document or taken photos of document ,Please choose document or take picture of document it before submiting the form by clicking Ok"
      );
      return <Redirect to="/docs" />;
    }
    return (
      <div id="DocsMain">
        <div id="DocsBox">
          <card>
            <cardcontent id="DocsHeading">
              <h3 id="FinalPageHeading" class="text-center default-text">
                Preview Page
              </h3>
            </cardcontent>
          </card>

          <hr style={{ width: "70%" }} />
          <card>
            <cardcontent>
              <div>
                <strong id="StrongText">Name:</strong>
                {this.state.name}
              </div>
              <div>
                <strong id="StrongText">Date Of Birth:</strong>{" "}
                {this.state.dateofbirth}
              </div>
              <div>
                <strong id="StrongText">Gender:</strong>
                {this.state.gender}
              </div>
            </cardcontent>
          </card>

          <cardcontent id="DocsHeading">
            <h3 id="FinalPageHeading" class="text-center default-text">
              Selfie:
            </h3>
          </cardcontent>
          <div>
            <img src={this.state.img} id="DocsImage" alt="Selfie"></img>
          </div>
          <hr style={{ width: "80%" }} />
        </div>

        <div id="DocsBox">
          <div id="DocsSelfieHeading">
            {" "}
            <strong id="StrongText">Goverment ID:</strong>
            {this.state.id}
          </div>
          <hr style={{ width: "60%" }} />

          <div>
            <img
              src={this.state.imgfront}
              id="DocsImage"
              alt="front side of document"
            ></img>
          </div>
          <div>Front side Image of {this.state.id}</div>

          <div>
            <img
              src={this.state.imgback}
              id="DocsImage"
              alt="back side of document"
            ></img>
          </div>
          <div>Back side Image of {this.state.id}</div>

          <div>
            <Button
              id="FinalPageSubmit"
              style={{ backgroundColor: "green" }}
              fullWidth
              type="submit"
              name="submit"
              onClick={this.handleonsubmit}
            >
              SUBMIT
            </Button>
          </div>
        </div>
      </div>
    );
  }
}
export default withRouter(Preview);
