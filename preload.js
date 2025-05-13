const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  selectFile: () => ipcRenderer.invoke('select-file'),
  runPythonScript: (videoPath) => ipcRenderer.invoke('run-python-script', videoPath)
}); 