const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  selectFile: () => ipcRenderer.invoke('select-file'),
  selectDirectory: () => ipcRenderer.invoke('select-directory'),
  runPythonScript: (videoPath, options) => ipcRenderer.invoke('run-python-script', videoPath, options),
  cancelProcessing: () => ipcRenderer.invoke('cancel-processing'),
  onPythonOutput: (callback) => {
    ipcRenderer.on('python-output', callback);
  },
  removePythonOutputListener: (callback) => {
    ipcRenderer.removeListener('python-output', callback);
  }
}); 