const express = require("express")
const app  = express()

app.use(function(req, res, next){
    console.log(`${new Date()} - ${req.method} request for ${req.url}`);
    next()
})

app.use(express.static("./static"))

app.get('/', (req, res)=>{
    res.sendFile(__dirname +"/static/classify.html")
})


app.listen(3000, function(){
    console.log("Server running on port 3000");
})
