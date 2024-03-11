console.log('I am a simple C4J client');
const net = require('net');
const MessageFramer = require('message-framer');
const JsonRpcProtocol = require('./json-rpc-protocol');
const { spawn } = require('child_process');


function findPower(resolution) {
    // Find the smallest exponent that, when raised to the power of two, is greater than or equal to the resolution
    return Math.ceil(Math.log2(resolution)); 
}

async function drawHorizontalLine(currentPower, maxPower, getItems) {
    try {
        const n = 2**currentPower
       const result = await getItems({ 
            mode: "check",
            xSize: n,
            ySize: 2*2**maxPower,
            xShift: n-1,
            yShift: 0,
            sync: true
        });
       console.log('result', result)
    } catch(exception) {
        console.error(exception);
    }
}

async function drawVerticalLine(currentPower, maxPower, getItems) {
    try {
        const n = 2**currentPower
       const result = await getItems({ 
            mode: "check",
            xSize: 2*2**maxPower,
            ySize: n,
            xShift: n-1,
            yShift: 0,
            sync: true
        });
       console.log('result', result)
    } catch(exception) {
        console.error(exception);
    }
}

async function capturePhoto(args) {
 try {   
    const process = 'C:\\Users\\MPaulson\\source\\repos\\TokyoDrift\\Acquisition\\Acquisitiond_v143.exe'
        await new Promise((resolve, reject) => {
            const childProcess = spawn(process, args);

            // Listen for data from the subprocess stdout
            // childProcess.stdout.on('data', (data) => {
            //     console.log(`stdout: ${data}`);
            // });

            // Listen for data from the subprocess stderr
            childProcess.stderr.on('data', (data) => {
                console.error(`stderr: ${data}`);
            });

            // Listen for the subprocess exit event
            childProcess.on('close', (code) => {
                console.log(`child process exited with code ${code}`);
                resolve(); // Resolve the promise once the subprocess has exited
            });

            // Listen for errors in spawning the child process
            childProcess.on('error', (err) => {
                console.error(`Failed to start child process: ${err}`);
                reject(err); // Reject the promise if an error occurs
            });
        });
    }

    catch (error) {
    console.error('Error running child process:', error);
    throw error; // Rethrow the error to be caught by the caller
}
}

// check that output directory for images has been included
console.log(args);
if (args.length !=3) {
    console.error('incorrect # of arguments, must provide image output directory')
} 

const outputFolder = args[2];


// open a tcp connection to the server with the given hostname and port
var client = new net.Socket();
var hostname = '169.254.159.128';
var port = 3003;

client.connect(port, hostname, async function() {
    // Display successful connection message.
    console.log('Connected to: ' + hostname + ':' + port);

    const rpc = new JsonRpcProtocol(client);
 
    // base the number of generated patterns on the display resolution of the projector
    const result = await rpc['status:getItems']({group: 'CONF'})
    const resolution = (result.find(item => item.desc == 'Output Resolution')).value
    console.log(resolution.value);

    const parts = resolution.split('x'); // Split the string by 'x'

    let horizontalRes = null;
    let verticalRes = null;

    if (parts.length === 2) {
        horizontalRes = parseInt(parts[0], 10); // horizontal resolution
        verticalRes = parseInt(parts[1], 10); // vertical resolution
        console.log("horizontal:", horizontalRes);
        console.log("vertical:", verticalRes);
    } else {
        console.error("Invalid resolution format");
    }

    const horizontalPower = findPower(horizontalRes);
    console.log('horizontal power', horizontalPower)
 
    const verticalPower = findPower(verticalRes);
    console.log('vertical power', verticalPower)

    const getItems = rpc['checks:drawPattern'];
   
    // declare output folder for images- pass as argument from python when called instead?
    //const outputFolder = 'C:\\Acquisition_Data\\debug'

    const delay = ms => new Promise(res => setTimeout(res, ms));
    await delay(6000);
    
    // Draw progressively smaller horizontal lines
    for (let i=1; i< horizontalPower+1; i++) {
        console.log('pattern num:',i)
        const delay = ms => new Promise(res => setTimeout(res, ms));
        try { // draw and display pattern
            //await delay(1000);
            await drawHorizontalLine(i, horizontalPower, getItems);
            console.log('drawHorizontalLine completed');
        } catch (error) {
            console.error('Error:', error);
        }

        try { // capture current pattern
            //await delay(1000);
            await capturePhoto([`pattern_y_${i}`, outputFolder]); //name corresponding to pattern #, need to provide output location
            console.log('capturePhoto completed');
        } catch (error) {
            console.error('Error:', error);
        }  
    }

   

    console.log('horizontal patterns completed');

    // Draw progressively larger vertical lines
    for (let i=1; i< verticalPower+1; i++) {
        const delay = ms => new Promise(res => setTimeout(res, ms));
        try { // draw and display pattern;
            await drawVerticalLine(i, verticalPower, getItems);
            console.log('drawVerticalLine completed');
        } catch (error) {
            console.error('Error:', error);
        }

        try { // capture current pattern
            await capturePhoto([`pattern_x_${i}`, outputFolder]); //name corresponding to pattern #, need to provide output location
            console.log('capturePhoto completed');
        } catch (error) {
            console.error('Error:', error);
        }
    }

    console.log('vertical patterns completed');

    rpc['checks:clearPattern']([{sync: true}]);

    process.exit(0);
});

