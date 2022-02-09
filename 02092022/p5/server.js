const express = require('express');
const cors = require('cors');
const http = require('http');
const path = require('path');


// Set up basic server stuff
const app = express();
app.use(cors());
const server = http.Server(app);


// Serve our `index.html` file
app.get('*', (req, res) => {
  const pathToHtml = path.resolve(__dirname, './index.html')
  res.sendFile(pathToHtml);
});

// --- Start our server listening on port 8000 --- //

server.listen(8000, () => {
  console.log('Server listening on http://localhost:8000/');
});
