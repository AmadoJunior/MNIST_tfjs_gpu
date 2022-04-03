//Dependencies
const express = require("express");
const app = express();



//Server HTML
app.use(express.static(__dirname + "/public/"));
app.get(/.*/, (req, res) => {
    res.sendFile(__dirname + "/public/index.html");
})

//POST Canvas Data
app.post('/', (req, res) => {
    res.send('Hello World!');
});

//Listen on PORT 5000
app.listen(5000, () => {
    console.log(`Listening on PORT: 5000`);
});