// .........Page 2 code
// ......
// ...redirecting not done

import React from "react";
import history from "../history";
import "../Css/Page1.css";
import "../Css/Page2.css";

import TextField from "@material-ui/core/TextField";
import Button from "@material-ui/core/Button";
import Typography from "@material-ui/core/Typography";
// Below belongs to only page 2----------------
//--------------------

import custo from "../Assets/customer.jpg";
const logOut = () => {
  localStorage.removeItem("auth");
  history.push("/");
};
var CryptoJS = require("crypto-js");
function detec() {
  if (
    navigator.userAgent.match(/Android/i) ||
    navigator.userAgent.match(/webOS/i) ||
    navigator.userAgent.match(/iPhone/i) ||
    navigator.userAgent.match(/iPad/i) ||
    navigator.userAgent.match(/iPod/i) ||
    navigator.userAgent.match(/BlackBerry/i) ||
    navigator.userAgent.match(/Windows Phone/i)
  ) {
  } else {
    alert(
      "We require particular functionality of mobile phones ,so please use mobile phone"
    );
    //history.push("/");
  }
}

function verifypass(event) {
  event.preventDefault();
  var Name = CryptoJS.AES.encrypt(
    JSON.stringify(sessionStorage.getItem("Name")),
    "my-secret-key@123"
  ).toString();
  var DOB = CryptoJS.AES.encrypt(
    JSON.stringify(sessionStorage.getItem("DOB")),
    "my-secret-key@123"
  ).toString();
  var gender = CryptoJS.AES.encrypt(
    JSON.stringify(sessionStorage.getItem("gender")),
    "my-secret-key@123"
  ).toString();
  sessionStorage.setItem("Name", Name);
  sessionStorage.setItem("DOB", DOB);
  sessionStorage.setItem("gender", gender);
  console.log(gender);
  detec();
  history.push("/selfie");
}
function FormDetailUser() {
  return (
    <div id="DetailsPageMain" className="main">
      {localStorage.getItem("auth") ? (
        <button onClick={logOut}>logout</button>
      ) : null}

      <h3 id="DetailsPageDefault" className="text-center default-text">
        Help us setup your account
      </h3>
      <p id="DetailsPageHeading" className="heading">
        <img id="cus" src={custo} alt="customer img"></img>
        We'll verify it with your KYC documents{" "}
      </p>

      <div className="App">
        <form className="form" onSubmit={verifypass}>
          <TextField
            variant="outlined"
            margin="normal"
            required
            fullWidth
            id="fullName"
            label="Your Full Name"
            name="fullName"
            autoComplete="fullName"
            autoFocus
            placeholder="eg.Tony Stark"
            onChange={(ev) => {
              sessionStorage.setItem("Name", ev.target.value);
            }}
          />

          <Typography id="DetailsPageCaption" variant="caption">
            Ensure it matches name on your PAN
          </Typography>

          <TextField
            variant="outlined"
            margin="normal"
            required
            fullWidth
            id="dob"
            label="Date of Birth"
            name="dob"
            autoComplete="dob"
            type="text"
            onFocus={(e) => (e.target.type = "date")}
            onBlur={(e) => (e.target.type = "text")}
            onChange={(ev) => {
              sessionStorage.setItem("DOB", ev.target.value);
            }}
          />

          <div
            id="genderElement"
            className="custom-control custom-radio custom-control-inline"
          >
            <input
              type="radio"
              className="custom-control-input"
              id="Male"
              name="inlineDefaultRadiosExample"
              required
              onChange={(ev) => {
                sessionStorage.setItem("gender", ev.target.id);
              }}
            />
            <label className="custom-control-label" >
              Male
            </label>
            <input
              type="radio"
              className="custom-control-input"
              id="Female"
              name="inlineDefaultRadiosExample"
              onChange={(ev) => {
                sessionStorage.setItem("gender", ev.target.id);
              }}
            />
            <label className="custom-control-label" >
              Female
            </label>
            <input
              type="radio"
              className="custom-control-input"
              id="Other"
              name="inlineDefaultRadiosExample"
              onChange={(ev) => {
                sessionStorage.setItem("gender", ev.target.id);
              }}
            />
            <label className="custom-control-label" >
              Others
            </label>
          </div>

          <Button
            id="submitButton"
            type="submit"
            fullWidth
            colour="primary"
            variant="contained"
          >
            CONTINUE
          </Button>
        </form>
      </div>
    </div>
  );
}

export default FormDetailUser;
