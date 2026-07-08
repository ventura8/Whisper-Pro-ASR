const fs = require("fs");
const path = require("path");
const vm = require("vm");

function loadScriptInContext(scriptPath, contextExtras = {}) {
  const absolutePath = path.resolve(scriptPath);
  const scriptCode = fs.readFileSync(absolutePath, "utf8");

  const context = vm.createContext({
    console,
    setTimeout,
    clearTimeout,
    setInterval,
    clearInterval,
    Date,
    Math,
    JSON,
    Promise,
    ...contextExtras,
  });

  vm.runInContext(scriptCode, context, { filename: absolutePath });
  return context;
}

function createMatchMediaStub(isDark = false) {
  return () => ({
    matches: isDark,
    addEventListener: () => {},
    removeEventListener: () => {},
  });
}

function evalInContext(context, expression) {
  return vm.runInContext(expression, context);
}

module.exports = {
  loadScriptInContext,
  createMatchMediaStub,
  evalInContext,
};
