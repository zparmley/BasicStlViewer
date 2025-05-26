from pathlib import Path
import tkinter

import dearpygui.dearpygui as dpg  # type: ignore
from dpgcontainers.containers import FileDialog
from dpgcontainers.containers import FileExtension
from dpgcontainers.containers import Menu
# from dpgcontainers.containers import Window
from dpgcontainers.containers import Button
from dpgcontainers.containers import ViewportMenuBar
from dpgmagictag.magictag import MagicTag

from stlviewer.gui.viewer import Viewer


class GUI:
    def __init__(self, *stls: Path):
        self.stls = stls
        self.tag = MagicTag('GUI')
        self.renderables = []
        self.menu = Menu(label='File')(
            Button(label='Open', callback=self.handle_menu_file_open),
        )

        self.renderables.append(ViewportMenuBar()(self.menu))
        self.file_dialog = FileDialog(
            directory_selector=False,
            tag='file_dialog',
            callback=self.handle_file_open,
            show=not stls,
            height=300,
        )(
            FileExtension('.stl', custom_text='[stl]'),
        )
        self.renderables.append(self.file_dialog)


    def get_screen_size(self):
        root = tkinter.Tk()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return width, height


    def start(self, debug: bool = False):
        width, height = self.get_screen_size()
        dpg.create_context()
        dpg.create_viewport(width=width, height=height)
        dpg.setup_dearpygui()

        self.render()
        if debug:
            dpg.show_item_registry()

        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()


    def render(self):
        for renderable in self.renderables:
            renderable.render(tag_prefix=self.tag)
        self.open_stls(*self.stls)


    def open_stls(self, *paths: Path):
        for path in paths:
            viewer = Viewer(path)
            viewer.render()

    ## Callbacks
    def handle_menu_file_open(self, sender, app_data, user_data):
        self.file_dialog.configure(show=True)

    def handle_file_open(self, sender, app_data, user_data):
        selections = app_data['selections']
        paths = map(Path, selections.values())
        self.open_stls(*paths)
