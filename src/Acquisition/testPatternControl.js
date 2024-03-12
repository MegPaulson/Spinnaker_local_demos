console.log('I am a simple C4J client');
const net = require('net');
const MessageFramer = require('message-framer');
const JsonRpcProtocol = require('./json-rpc-protocol');
const { spawn } = require('child_process');
const { default: test } = require('node:test');
//const fetch = require('node-fetch');
const fetch = (...args) => import('node-fetch').then(({default: fetch}) => fetch(...args));
const fs = require('fs').promises;

async function readFileAsBuffer(filePath) {
    const fileContent = await fs.readFile(filePath);
    return fileContent;
}

// sample specified tempature value and write to output file
async function sampleTemperature(rpc, outputFile, tempItems) {

    // tempitems = array of tempature items to sample based on name given by group.item.desc, ex. ['TEMP1', 'TEMP2', 'TEMP3']

    const result = await rpc['status:getItems']({group: 'TEMP'});

    for (item in tempItems) {
        const temp = result.find(item => item.desc == item)
        console.log(temp);
        const timestamp = new Date().toISOString();
        const content = `${timestamp} ${item.desc} ${temp.value}\n`;
        // write to file with timestamp
        await fs.appendFile(outputFile, content);
    }
}

async function capturePhoto(args) {
    try {   
       const process = 'C:\\Users\\MPaulson\\source\\repos\\TokyoDrift\\Acquisition\\Acquisitiond_v143.exe'
           await new Promise((resolve, reject) => {
            // usage: args = [imageName, outputFolder]
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

// upload a local test pattern
async function uploadLocalPattern(imageFile, rpc) {

    const uploadPath = await rpc['testPattern:getUploadPath']({});

    console.log(uploadPath);

    var client = new net.Socket();
    var hostname = '169.254.159.128';
    var port = 80; // http post is on port 80

    return new Promise((resolve, reject) => {
    const result = client.connect(port, hostname, imageFile, uploadPath, async function(err, data) {

        console.log('Connected to: ' + hostname + ':' + port);

        const rpc = new JsonRpcProtocol(client);
    
        // need device name- ip?
        const device = `169.254.159.128`;
        const host = `http://${device}/`;
        const httpPath = host + uploadPath;

        console.log(httpPath);

        const fileContent = await readFileAsBuffer(imageFile);

        const fileBlob = new Blob([fileContent]);
        const formData = new FormData();
        formData.append('file', fileBlob, imageFile);
 
        //let data;
        try {const response = await fetch(httpPath, {
            method: 'POST',
            body: formData
            })
            if (!response.ok) {
                reject(err);
                throw new Error('response not ok');

            }

            else {
            data = await response.json();
            console.log("SrcName:",data.result.token); // SrcName to be used as param for testPattern:display
            resolve(data);
            //return data; //data.result.token;
            }
        }
        catch(error) {
            console.error('Error:', error);
            throw error;
        }

        client.end();
    });
});

}


// display an uploaded test pattern without saving
async function displayUploadedPattern(imageName, display) {
    try {
        console.log("imName",imageName)
        const result = await display({
            type: "remote",
            srcName: `${imageName}`,
            format: "PNG",
            sync: true
        });
        return result;
    }
    catch(exception) {
        console.error("Error:", exception);
    }
    
}

    /*  function to run a test of length testPeriod- a user-defined time intervel in minutes, 
    where focus and temperature are sampled at the user-defined intervals focusSamplingInterval and
    tempSamplingInterval (in minutes) respectively. Header amd tail photos are captured and saved
    (no focus data is saved for these photos).
    */
async function runTest(testPeriod, focusSamplingInterval, tempSamplingInterval, rpc, display, tempItems, dataOutputFile, imageOutputFolder, defaultPattern, testPattern) { 
    try {
        const startTime = new Date();
        const endTime = new Date(startTime.getTime() + testPeriod*60000);
        const fs = new Date(startTime.getTime() + focusSamplingInterval*60000);
        const ts = new Date(startTime.getTime() + tempSamplingInterval*60000);

        // begin test

        // display test pattern
        await displayUploadedPattern(testPattern, display);
        //capture and save header photo
        await capturePhoto(imageOutputFolder);
        // display default pattern
        await displayUploadedPattern(defaultPattern, display);

        let tempInterval, focusInterval;

        // Start the intervals
        tempInterval = setInterval(async () => {
            try {
                await sampleTemperature(rpc, dataOutputFile, tempItems);
            } catch (error) {
                console.error("Error in tempInterval:", error);
            }
        }, ts);
        
        focusInterval = setInterval(async () => {
            try {
                // display test pattern
                await displayUploadedPattern(testPattern, display);
                // capture photo, write focus data to dataOutputFile
                await capturePhoto(dataOutputFile);
                // display default pattern
                await displayUploadedPattern(defaultPattern, display);
            } catch (error) {
                console.error("Error in focusInterval:", error);
            }
        }, fs);

        // Stop the intervals when the period is over
        setTimeout(() => {
            try {
                clearInterval(tempInterval);
                clearInterval(focusInterval);
            } catch (error) {
                console.error("Error in setTimeout:", error);
            }
        }, endTime - Date.now());

        await displayUploadedPattern(testPattern, display);
        //capture and save tail photo
        await capturePhoto(imageOutputFolder);
        // display default pattern
        await displayUploadedPattern(defaultPattern, display);

        // end test
        
    }
    catch(exception) {
        console.error("Error:", exception);
    }
}



var client = new net.Socket();
var hostname = '169.254.159.128';
var port = 3003;

client.connect(port, hostname, async function() {
    // Display successful connection message.
    console.log('Connected to: ' + hostname + ':' + port);

    // create rpc object
    const rpc = new JsonRpcProtocol(client);

    // swap between uploaded black test pattern and the single pixel / cross pattern generated by calibration process, and synchronize with capture
    // need to upload both or draw one if unable to control built in patterns.. 

    const display = rpc['testPattern:display'];
    const testimage = "C:\\Acquisition_Data\\Gray_code_patterns4\\pattern_x_0.png";
    //const testimage = "C:\\Users\\MPaulson\\Downloads\\postTestPattern.png";
    const defaultPattern =  "C:\\Acquisition_Data\\crosshairs\\crosshairs.png";

    // uploaded source file names- will be populated with response from upload POST request
    let testPatternName;
    let defaultPatternName;

    // upload test pattern
    uploadLocalPattern(testimage, rpc)
        .then(data => {
            // if upload is successful..
            console.log('Received data:', data.result.token);
            testPatternName = data.result.token;
        })
        .catch(error => {
            console.error('Error:', error);
            });
    
    // upload default pattern
    uploadLocalPattern(defaultPattern, rpc)
        .then(data => {
            // if upload is successful..
            console.log('Received data:', data.result.token);
            defaultPatternName = data.result.token;
        })
        .catch(error => {
            console.error('Error:', error);
            });

    // see if projector is in test-ready state- shutter open, max brightness, etc.
    //const open = await rpc['shutter:open']({});
    //const brightness = await rpc['brightness:set']({value: 100});

    // run test
    const testPeriod = 10; // in minutes
    const focusSamplingInterval = 1; // in minutes
    const tempSamplingInterval = 1; // in minutes
    const tempItems = ['TEMP1', 'TEMP2', 'TEMP3'];
    const dataOutputFile = 'C:\\Acquisition_Data\\debug\\tempData.txt';
    const imageOutputFolder = 'C:\\Acquisition_Data\\debug';
    runTest(testPeriod, focusSamplingInterval, tempSamplingInterval, rpc, display, tempItems, dataOutputFile, imageOutputFolder, defaultPatternName, testPatternName);
    
    // test complete
    console.log('Test complete');

    // close connection
    client.end();
    console.log('Connection closed');
    // process.exit(0);
});