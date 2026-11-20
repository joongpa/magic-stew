import { mount } from "svelte";
import App from "./App.svelte";
import "./style.css";
import {startOfflineCache} from './lib/offline-cache';

if (import.meta.env.PROD) {
  void startOfflineCache(new URL(import.meta.env.BASE_URL, location.origin).href,
    document.querySelector<HTMLMetaElement>('meta[name="app-release"]')?.content);
}

mount(App, { target: document.getElementById("app")! });
