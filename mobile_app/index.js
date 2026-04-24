import React from 'react';
import { App as ExpoApp } from 'expo';
import App from './App';

const expoApp = new ExpoApp();

export default expoApp.registerComponent(App);