// See https://www.jsonrpc.org/specification

const MessageFramer = require('message-framer');

const replaceUndefinedWithNull = value => (typeof value === 'undefined') ? null : value;

function trace() {
  // OPTIONAL: you can log messages messages here.
  // console.log(...arguments);
}

class JsonRpcProtocol {
  constructor(socket, target) {

    let nextUnusedMessageId = 1;
    let mf = new MessageFramer();
    let handlers = {};
    let promises = {};
    socket.on('data', mf.add.bind(mf));
    const send = json => {
      trace('Sending RPC message', json);
      socket.write(json);
      socket.write('\0');
    }
    const read = json => {
      trace('Received RPC message', json);
    }

    mf.on('message', buffer => {
      if (buffer.length == 0) return; // ignore empty buffers
      let json = buffer.toString('utf8');
      read(json); // This is only for tracing RPC messages
      let rpcmsg;
      try {
        rpcmsg = JSON.parse(json);
      } catch(err) {
        console.warn('Ignoring non-JSON rpc message', err, json);
        return;
      }
      if (typeof rpcmsg !== 'object') return;
      if (rpcmsg.hasOwnProperty('method')) {
        // It's a request.
        let rpcMethods = target || {};
        let fn = rpcMethods[rpcmsg.method];
        if (typeof (fn) === 'function') {
          // TODO: Make sure we're allowed to call this method.
          if (promises[rpcmsg.id]) {
            // Protocol violation: the remote is using an ID that conflicts with a call in progress.
            // TODO: Protocol violations should reject the misbehaving connection.
            // HACK: Tolerate it by dropping this message.
            return;
          } else {
            let response = {
              jsonrpc: '2.0',
              id: rpcmsg.id
            };
            promises[rpcmsg.id] = fn(rpcmsg.params)
              .then(result => {
                if (promises[rpcmsg.id]) {
                  delete promises[rpcmsg.id];
                  // Important! Many types will not survive the trip to JSON and back,
                  // most notably the return value of `undefined` cannot be converted to JSON
                  // but the field `result` is REQUIRED by the JSONRPC spec.
                  response.result = replaceUndefinedWithNull(result);
                  send(JSON.stringify(response));
                }
              })
              .catch(error => {
                if (promises[rpcmsg.id]) {
                  // This is useful for debugging:
                  // If the RPC target throws, you can log it here:
                    // console.error(error);
                  delete promises[rpcmsg.id];
                  // Important! Many types will not survive the trip to JSON and back,
                  // most notably the return value of `undefined` cannot be converted to JSON
                  // but the field `error` is REQUIRED by the JSONRPC spec.
                  response.error = replaceUndefinedWithNull(error);
                  send(JSON.stringify(response));
                }
              });
          }
        } else {
          // The RPC target is missing the requested method,
          let response = {
            jsonrpc: '2.0',
            id: rpcmsg.id,
            error: { code: -32601, message: `Method '${rpcmsg.method}' not found` }
          };
          console.error(response);
          send(JSON.stringify(response));
        }
      } else {
        // It's a response.
        let handler = handlers[rpcmsg.id];
        if (handler) {
          delete handlers[rpcmsg.id];
          if (rpcmsg.error) {
            handler.reject(rpcmsg.error);
          } else {
            handler.resolve(rpcmsg.result);
          }
        } else {
          // Protocol violation: unsolicited response.
          // TODO: Protocol violations should reject the misbehaving connection.
          // HACK: ignore the unsolicited reponse.
          return;
        }
      }
    });

    // TODO: if the socket gets closed then all pending calls need to get rejected
    // with RPC connection errors.
    socket.on('close', () => {
      for(let id in promises) {
        let promise = promises[id];
        delete promises[id];
        let response = {
          jsonrpc: '2.0',
          id,
          error: { code: -32300, message: `Connection closed` }
        };
        send(JSON.stringify(response));
      }
    });


    // Proxy objects are neat:
    // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Proxy
    return new Proxy(this, {
      get: function (obj, prop /*, receiver*/) {
        return function () {
          let rpcmsg = {
            jsonrpc: '2.0',
            id: nextUnusedMessageId++,
            method: prop,
            params: arguments[0]
          };
          send(JSON.stringify(rpcmsg));

          return new Promise((resolve, reject) => {
            handlers[rpcmsg.id] = { resolve, reject };
          });
        };
      }
    });
  }
};

module.exports = JsonRpcProtocol;