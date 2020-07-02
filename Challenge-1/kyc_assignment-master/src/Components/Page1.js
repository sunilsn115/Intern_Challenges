import React, { useState, useEffect } from "react";

import "../Css/Page1.css";
import history from "../history";
import TextField from "@material-ui/core/TextField";
import Button from "@material-ui/core/Button";
import FormControlLabel from "@material-ui/core/FormControlLabel";
import Checkbox from "@material-ui/core/Checkbox";
import Typography from "@material-ui/core/Typography";
import loginLogo from "../Assets/login.jpg";
function FormUserAuth(props) {
  const [name, setname] = useState([]);
  const [password, setpassword] = useState([]);
  const [dark, setDark] = useState([false]);
  useEffect(() => {
    localStorage.getItem("dark") ? setDark(true) : setDark(false);
  }, []);
  const verifypass = (event) => {
    event.preventDefault();

    fetch(`https://finalkyc.free.beeceptor.com`)
      // We get the API response and receive data in JSON format...
      .then((response) => response.json())
      // ...then we update the users state
      .then((data) => {
        console.log(data);
        let c = 0;
        for (let i = 0; i < data.users.length; i++) {
          if (
            data.users[i].name === name &&
            data.users[i].password === password
          ) {
            localStorage.setItem("auth", "app");
            history.push("/detail");

            c = 1;
            break;
          }
        }
        if (c !== 1) {
          alert("wrong credentials");
          window.location.reload();
        }
      })
      .catch((error) => console.log(error));
  };
  const handleChangeName = (e) => {
    setname(e.target.value);
  };
  const HandlePassword = (e) => {
    setpassword(e.target.value);
  };
  return (
    <div id="LoginPageMain" className={dark ? "main-dark" : "main"}>
      <h3 className="text-center default-text">Hi! Welcome to ZestMoney</h3>
      <h5 className="text-center default-text">One stop KYC solution</h5>
      <img id="loginLogo" src={loginLogo} alt="customer img"></img>
      <Typography id="signInHeading" component="h1" variant="h5">
        <strong>Sign in</strong>
      </Typography>
      <div className="App">
        <div className="form">
          <TextField
            variant="outlined"
            margin="normal"
            required
            fullWidth
            id="email"
            label="UserName"
            name="username"
            autoComplete="username"
            autoFocus
            placeholder="Tony@abc"
            onChange={handleChangeName}
          />
          <TextField
            variant="outlined"
            margin="normal"
            required
            fullWidth
            id="password"
            label="Password"
            name="password"
            autoComplete="password"
            // autoFocus
            type="password"
            onChange={HandlePassword}
          />
          <FormControlLabel
            control={<Checkbox value="remember" color="primary" />}
            label="Remember me"
          />
          <Button
            id="LoginPageButton"
            className="Tap"
            type="submit"
            fullWidth
            colour="sucess"
            variant="contained"
            onClick={verifypass}
          >
            GET OTP
          </Button>
        </div>
      </div>
    </div>
  );
}

export default FormUserAuth;
