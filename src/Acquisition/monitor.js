console.log('I am a simple C4J client');
const net = require('net');
const MessageFramer = require('message-framer');
const JsonRpcProtocol = require('./json-rpc-protocol');

// open a tcp connection to the server with the given hostname and port
var client = new net.Socket();
var hostname = '169.254.159.128';
var port = 3003;
client.connect(port, hostname, async function() {
    // Display successful connection message.
    console.log('Connected to: ' + hostname + ':' + port);

    const rpc = new JsonRpcProtocol(client);

    setInterval(async () => {
        try {
            const result = await rpc['status:getItems']({group: 'TEMP'});
            const temp = result.find(item => item.idx == 4)
            console.log(temp);
        } catch(ex) {
            console.error(ex);
        }
    }, 2000);
        // const items2 = await rpc['status:getItems']({group: 'COOL'});
    // console.log(toTableString(items2, ['desc', 'value']));

});
