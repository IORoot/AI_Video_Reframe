const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const { PythonShell } = require('python-shell');

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    }
  });

  mainWindow.loadFile('index.html');
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// Handle file selection
ipcMain.handle('select-file', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [
      { name: 'Videos', extensions: ['mp4', 'mov', 'avi'] }
    ]
  });
  
  if (!result.canceled && result.filePaths.length > 0) {
    return result.filePaths[0];
  }
  return null;
});

// Handle Python script execution
ipcMain.handle('run-python-script', async (event, videoPath, options = {}) => {
  return new Promise((resolve, reject) => {
    // Generate output path by adding '_processed' before the extension
    const outputPath = videoPath.replace(/\.[^/.]+$/, '_processed$&');
    
    // Build command line arguments array
    const args = [
      '--input', videoPath,
      '--output', outputPath
    ];

    // Add all optional parameters if they are provided
    if (options.target_ratio !== undefined) args.push('--target_ratio', options.target_ratio.toString());
    if (options.max_workers !== undefined) args.push('--max_workers', options.max_workers.toString());
    if (options.detector !== undefined) args.push('--detector', options.detector);
    if (options.skip_frames !== undefined) args.push('--skip_frames', options.skip_frames.toString());
    if (options.conf_threshold !== undefined) args.push('--conf_threshold', options.conf_threshold.toString());
    if (options.model_size !== undefined) args.push('--model_size', options.model_size);
    if (options.object_classes !== undefined) args.push('--object_classes', ...options.object_classes.map(x => x.toString()));
    if (options.track_count !== undefined) args.push('--track_count', options.track_count.toString());
    if (options.padding_ratio !== undefined) args.push('--padding_ratio', options.padding_ratio.toString());
    if (options.size_weight !== undefined) args.push('--size_weight', options.size_weight.toString());
    if (options.center_weight !== undefined) args.push('--center_weight', options.center_weight.toString());
    if (options.motion_weight !== undefined) args.push('--motion_weight', options.motion_weight.toString());
    if (options.history_weight !== undefined) args.push('--history_weight', options.history_weight.toString());
    if (options.saliency_weight !== undefined) args.push('--saliency_weight', options.saliency_weight.toString());
    if (options.face_detection) args.push('--face_detection');
    if (options.weighted_center) args.push('--weighted_center');
    if (options.blend_saliency) args.push('--blend_saliency');
    if (options.apply_smoothing) args.push('--apply_smoothing');
    if (options.smoothing_window !== undefined) args.push('--smoothing_window', options.smoothing_window.toString());
    if (options.position_inertia !== undefined) args.push('--position_inertia', options.position_inertia.toString());
    if (options.size_inertia !== undefined) args.push('--size_inertia', options.size_inertia.toString());
    if (options.debug) args.push('--debug');

    const pythonOptions = {
      mode: 'text',
      pythonPath: path.join(__dirname, 'venv', 'bin', 'python'),
      pythonOptions: ['-u'], // unbuffered output
      scriptPath: __dirname,
      args: args
    };
    
    // Debug: log Python execution options
    console.log('Running Python script with options:', pythonOptions);
    
    PythonShell.run('main.py', pythonOptions).then(messages => {
      // Debug: log completion messages
      console.log('Python script completed. Messages:', messages);
      resolve({ 
        success: true, 
        messages,
        outputPath: outputPath 
      });
    }).catch(err => {
      // Debug: log errors from PythonShell
      console.error('Error running Python script:', err);
      // Format the error object to ensure it can be properly serialized
      const errorMessage = err.message || err.toString();
      const errorDetails = {
        success: false,
        error: errorMessage,
        stack: err.stack || '',
        code: err.code || 'UNKNOWN_ERROR'
      };
      reject(errorDetails);
    });
  });
}); 