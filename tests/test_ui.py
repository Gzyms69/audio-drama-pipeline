import pytest
import gradio as gr
from audio_drama.ui.app import create_ui, find_available_epubs, get_existing_scene_files

def test_find_available_epubs():
    epubs = find_available_epubs()
    assert isinstance(epubs, list)
    assert len(epubs) > 0

def test_get_existing_scene_files():
    scenes = get_existing_scene_files()
    assert isinstance(scenes, list)

def test_create_ui():
    demo = create_ui()
    assert isinstance(demo, gr.Blocks)
    assert demo.title == "Audio Drama Studio"
