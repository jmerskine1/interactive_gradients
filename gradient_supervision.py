
import sys

from PyQt6.QtCore import Qt, QPoint, QThread

from PyQt6.QtGui import QPainter, QBrush, QPen, QColor, QIntValidator, QIcon, QStandardItemModel, QStandardItem

from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedLayout,
    QSlider,
    QScrollArea,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTableView,
    QVBoxLayout,
    QWidget,
    QSplitter
)


import pickle
import numpy as np
import matplotlib
matplotlib.use('QtAgg')

import pandas as pd

from custom_datasets import datasets

from utilities import generate_figure_gui

from utilities_pyqt import (InProgress, 
                            Color, 
                            LossMPL, 
                            ScatterMPL, 
                            TableModel, 
                            LearningAlgorithm, 
                            LearningThread,
                            KnowledgeList)

from utilities_pyqt import (display_gui, 
                            get_selected_cell,
                            update_dataset_config, 
                            load_results, 
                            store_results, 
                            overwrite_results)

from interactive_pipeline import Pipeline

session_id = "default_session"

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.session_id = session_id
        self.pipeline = Pipeline()
        self.learning_thread = LearningThread(self.pipeline)

        self.setWindowTitle("Gradient Supervision")

        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.TabPosition.North)
        tabs.setMovable(False)

        # GUI Variables
        self.title_height = 20
        # set the border color for both the table and its cells
        self.border_color = "#F8B195"
        # set the background color
        self.bg_color = "#364557"
        # Define 2D TAB
        # Left column | Dataset config and Knowledge Workbench


        self.create_data_widget()
        self.create_knowledge_widget()
        self.create_loss_widget()
        self.create_controls()
        self.create_scatter_widget()
        self.create_results()
        self.initialise_thread()



        """
        ASSEMBLE GUI
        """

        self.update_data_button.clicked.connect(self.handle_data_update)

        leftcol = QVBoxLayout()
        leftcol.addWidget(self.datasetWidget)
        leftcol.addWidget(self.knowledgeWidget)
        leftWidget = QWidget()
        leftWidget.setLayout(leftcol)
        leftWidget.setMaximumWidth(300)


        midcol = QVBoxLayout()
        midcol.addWidget(self.controlPanel)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setMinimumWidth(300)
        
        splitter.addWidget(self.lossWidget)
        splitter.addWidget(self.scatterWidget)
        
        
        splitter.setStyleSheet(           "QSplitter::handle {"
            "   background-color: gray;"
            "   border: 1px solid #333;"
            "   border-radius: 4px;"
            "}"
            "QSplitter::handle:hover {"
            "   background-color: lightgray;"
            "}")
    
        midcol.addWidget(splitter)

        midcol.setContentsMargins(0,0,0,0)
        midcol.setSpacing(0)

        rightcol = QVBoxLayout()
        rightcol.addWidget(self.scoreWidget)

        allcol = QHBoxLayout()
        allcol.addWidget(leftWidget)
        allcol.addLayout(midcol)
        allcol.addLayout(rightcol)
        _2DWidget = QWidget()
        _2DWidget.setLayout(allcol)

        tabs.addTab(_2DWidget,'2D')

        # DEFINE 3D+ TAB
        tabs.addTab(InProgress(), 'Sentiment Classification')
        tabs.addTab(InProgress(), '3D+')

        self.setCentralWidget(tabs)
    
    def handle_play_pause(self):
        if self.learning_thread.learning_algorithm.is_running():
            self.button_playpause.setIcon(QIcon('icons/play.png'))
            self.learning_thread.learning_algorithm.pause()

        else:
            self.learning_thread.start()
            self.button_playpause.setIcon(QIcon('icons/pause.png'))
    
    def initialise_thread(self):
        self.learning_thread = LearningThread(self.pipeline)
        self.learning_thread.learning_algorithm.stateUpdate.connect(self.state_update)

    def state_update(self,pipeline):
        self.pipeline = pipeline
        self.lossWidget.update_plot(self.pipeline)
        self.scatterWidget.update_plot(self.pipeline)


    def handle_data_update(self):
        if self.learning_thread.learning_algorithm.is_running():
            self.button_playpause.setIcon(QIcon('icons/play.png'))
            self.learning_thread.learning_algorithm.pause()


        if self.learning_thread.isRunning():
            self.learning_thread.quit()
        
        update_dataset_config(self.dataset_combobox.currentText(),int(self.size_input.text() or 0))
        QThread.msleep(100)
        self.pipeline = Pipeline()
        self.scatterWidget.update_plot(self.pipeline)
        self.lossWidget.reset(self.pipeline)
        self.lossWidget.update_plot(self.pipeline)
        self.list_tools_stack.setCurrentIndex(2)
        self.lossWidget.update_plot(self.pipeline)
        self.scatterWidget.update_plot(self.pipeline)
        self.initialise_thread()


    def handle_knowledge_update(self):
        vec = np.array([self.slider_convert(self.sliders['slider0'].value()),
                        self.slider_convert(self.sliders['slider1'].value())])
        mag = np.linalg.norm(vec)
        unit_vec = np.array(vec/mag)
        self.pipeline.add_knowledge(self.scatterWidget.selected_point,unit_vec)
        self.scatterWidget.update_plot(self.pipeline)

    def slider_convert(self,value):
        return (value-50)/50
    
    def create_data_widget(self):
        """
        DATASET: TYPE AND SIZE
        """
        # Create the dataset label and dropdown box
        dataset_label = QLabel("Dataset")
        dataset_label.setMaximumWidth(50)
        self.dataset_combobox = QComboBox()
        self.dataset_combobox.setMinimumWidth(100)
        # for key in datasets:
        self.dataset_combobox.addItems(datasets.keys())
        
        self.dataset_combobox.setCurrentText(self.pipeline.data_params['dataset'])
        
        # self.dataset_combobox.addItem("CIFAR-10")
        data_type = QHBoxLayout()
        data_type.addWidget(dataset_label)
        data_type.addWidget(self.dataset_combobox)
        type_widget = QWidget()
        type_widget.setLayout(data_type)
        

        # Create the size label and dropdown box
        size_label = QLabel("Size")
        self.size_input = QLineEdit()
        # self.size_input.setPlaceholderText('Enter a value...')
        self.size_input.setText(str(self.pipeline.data_params['size']))
        self.size_input.setValidator(QIntValidator())
        data_size = QHBoxLayout()
        data_size.addWidget(size_label)
        data_size.addWidget(self.size_input)
        size_widget = QWidget()
        size_widget.setLayout(data_size)

        # Create update data button
        self.update_data_button = QPushButton("Update")
    
        datasetLayout = QVBoxLayout()
        datasetTitle = QLabel("<center><b>Data</b></center>")
        datasetTitle.setMaximumHeight(self.title_height)
        datasetLayout.addWidget(datasetTitle)
        datasetLayout.addWidget(type_widget)
        datasetLayout.addWidget(size_widget)
        datasetLayout.addWidget(self.update_data_button)


        self.datasetWidget = QWidget()
        self.datasetWidget.setLayout(datasetLayout)
        self.datasetWidget.setAutoFillBackground(True)
        self.datasetWidget.setMaximumHeight(200)



    def create_knowledge_widget(self):

        """
        KNOWLEDGE WORKBENCH: FEATURE IMPORTANCES FOR X & Y
        """
        knowledgeLayout = QVBoxLayout()
        knowledgeTitle = QLabel("<center><b>Annotation Toolbox</b></center>")
        knowledgeTitle.setMaximumHeight(self.title_height)
        
        # 'add knowledge' button
        self.button_add_knowledge = QPushButton()
        self.button_add_knowledge.setIcon(QIcon('icons/add.png'))
        
        # create list of current knowledge with deletable items
        self.knowledgeList = KnowledgeList()
        
        knowledgetoolbar = QVBoxLayout()
        knowledgetoolbar.addWidget(self.button_add_knowledge)

        listorprompt = QStackedLayout()
        listorprompt.addWidget(QLabel("""<center><i>There is no knowledge for this data point 
                                        <br> <br> Click the '+' icon to add some.</i></center>""",wordWrap=True))
        listorprompt.addWidget(self.knowledgeList)
        listorprompt.setCurrentIndex(0)

        self.listpromptWidget = QWidget()
        self.listpromptWidget.setLayout(listorprompt)

        knowledgetoolbar.addWidget(self.listpromptWidget)
        knowledgeWidget = QWidget()
        knowledgeWidget.setLayout(knowledgetoolbar)
    
        
        featureLayout = QVBoxLayout()
        self.sliders = {}
        for feature in range(self.pipeline.train_dataset.data.X.shape[1]):
            label = QLabel(f"Feature {feature}")
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(100)
            slider.setTickInterval(50)
            slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            slider.setValue(50)
            slider.setObjectName(f"slider{feature}")
            self.sliders[f"slider{feature}"] = slider
            labelled_slider= QHBoxLayout()
            labelled_slider.addWidget(QLabel("-1"))
            labelled_slider.addWidget(slider)
            labelled_slider.addWidget(QLabel("1"))
            labelled_slider_widget = QWidget()
            labelled_slider_widget.setLayout(labelled_slider)

            titled_slider = QVBoxLayout()
            titled_slider.addWidget(label)
            titled_slider.addWidget(labelled_slider_widget)
            
            slider_widget = QWidget()
            slider_widget.setLayout(titled_slider)
            featureLayout.addWidget(slider_widget)
  

        self.button_done = QPushButton("Done")
        featureLayout.addWidget(self.button_done)
        featureTools = QWidget()
        featureTools.setLayout(featureLayout)
        
        self.list_tools_stack = QStackedLayout()
        self.list_tools_stack.addWidget(knowledgeWidget)
        self.list_tools_stack.addWidget(featureTools)
        self.list_tools_stack.addWidget(QLabel("<center><i>Click on a point to view <br> knowledge base</i></center>"))
        self.list_tools_stack.setCurrentIndex(2)
        list_tools_stack_widget = QWidget()
        list_tools_stack_widget.setLayout(self.list_tools_stack)
        
        knowledgeLayout.addWidget(knowledgeTitle)
        knowledgeLayout.addWidget(list_tools_stack_widget)
        self.knowledgeList.knowledgestack_index.connect(self.list_tools_stack.setCurrentIndex)
        self.button_add_knowledge.clicked.connect(lambda: self.list_tools_stack.setCurrentIndex(1))

        self.knowledgeWidget = QWidget()
        self.knowledgeWidget.setLayout(knowledgeLayout)
        self.knowledgeWidget.setAutoFillBackground(True)


    def create_loss_widget(self):
        """
        TRAINING: PLOT OF LOSS VS EPOCH
        """
        self.lossWidget = LossMPL(self.pipeline,self.session_id)
        # self.lossWidget.setMaximumHeight(400)
        self.lossWidget.update_plot(self.pipeline)

        

    def create_controls(self):
        """
        CONTROL BUTTONS: PAUSE/PLAY, RUN (UNTIL PAUSE OR FOR N EPOCHS), STOP & EVAL
        """
        self.button_playpause = QPushButton()
        self.button_playpause.setIcon(QIcon('icons/play.png'))
        self.play = True
        self.button_playpause.clicked.connect(self.handle_play_pause)

        self.button_save = QPushButton()
        self.button_save.setIcon(QIcon('icons/save.png'))
        self.button_save.clicked.connect(self.handle_save_results)
        
        self.button_delete = QPushButton()
        self.button_delete.setIcon(QIcon('icons/delete.png'))
        self.button_delete.clicked.connect(self.handle_delete_results)
        
        buttons = [self.button_playpause,self.button_save, self.button_delete]
        controlPanelLayout = QHBoxLayout()

        for button in buttons:
            button.setMaximumWidth(50)
            button.setMinimumHeight(50)
            controlPanelLayout.addWidget(button)

        self.controlPanel = QWidget()
        self.controlPanel.setLayout(controlPanelLayout)
    
    def create_scatter_widget(self):
        """
        DATA VISUALISATION: SCATTER DATA WITH PICAKBLE POINTS
        """
        self.scatterWidget = ScatterMPL(self.pipeline)
        self.scatterWidget.selectionSignal.connect(lambda: self.list_tools_stack.setCurrentIndex(0))
        self.scatterWidget.selectNoneSignal.connect(lambda: self.list_tools_stack.setCurrentIndex(2))
        
        self.sliders['slider0'].valueChanged[int].connect(lambda value:self.scatterWidget.live_vectors(
                                self.slider_convert(value), self.slider_convert(self.sliders['slider1'].value())))
        self.sliders['slider1'].valueChanged[int].connect(lambda value:self.scatterWidget.live_vectors(
                                self.slider_convert(self.sliders['slider0'].value()),self.slider_convert(value)))  
    
        self.button_done.clicked.connect(self.handle_knowledge_update)
        self.scatterWidget.update_plot(self.pipeline)
    
    def create_results(self):
        """
        SCORES: TABLES OF RESULTS
        """
        
        self.results_data = load_results(self.session_id)

        # create a table view and set its model to an empty item model
        self.table = QTableView(self)
        self.model = QStandardItemModel()
        self.table.setModel(self.model)


        scroll_bar_h = self.table.horizontalScrollBar()
        scroll_bar_v = self.table.verticalScrollBar()

        scroll_bar_h.setStyleSheet("QScrollBar:handle {background: #F8B195;}")
        scoreLayout = QVBoxLayout()
        scoreLayout.addWidget(QLabel("<center><b>Scores</b></center>"))
        scoreLayout.addWidget(self.table)
        # Set the aspect ratio mode to maintain a square aspect ratio

        # self.scoreWidget = SquareWidget()
        self.scoreWidget = QWidget()
        self.scoreWidget.setLayout(scoreLayout)
        self.scoreWidget.setMinimumWidth(330)
        self.scoreWidget.setMaximumWidth(400)
        self.update_table_view()
        
        self.table.clicked.connect(lambda: get_selected_cell(self.table))
    
    def handle_save_results(self):
        if self.learning_thread.learning_algorithm.is_running():
            self.button_playpause.setIcon(QIcon('icons/play.png'))
            self.learning_thread.learning_algorithm.pause()

        # create a message box and set its layout
        msg = QMessageBox()
        msg.setWindowTitle('Saving progress...')
        msg.setIcon(QMessageBox.Icon.Question)

        # add buttons to the message box
        user_name, ok = QInputDialog.getText(msg, "Name Input", "Enter your name:")
        
        # add buttons to the message box
        ok_button = QPushButton('Ok')
        msg.addButton(ok_button, QMessageBox.ButtonRole.AcceptRole)
        cancel_button = QPushButton('Cancel')
        msg.addButton(cancel_button, QMessageBox.ButtonRole.RejectRole)

        # display the message box and wait for a button to be clicked
        result = msg.exec()

        # handle the button click
        if ok_button.clicked:
            if user_name == None:
                errormsg = QMessageBox("Error: Enter a valid name (cannot be empty)")
                errormsg.addButton(ok_button,QMessageBox.ButtonRole.AcceptRole)
                errormsg.exec()
                if errormsg.clickedButton() == ok_button:
                    return

            else:
                store_results(self.session_id,user_name,self.pipeline)
                self.results_data = load_results(self.session_id)
                self.update_table_view()
                self.lossWidget.update_saved_plots(self.session_id)
                self.lossWidget.update()
        else:
            print("User cancelled the input")
            return
        
        
        
    def handle_delete_results(self):
        if self.learning_thread.learning_algorithm.is_running():
            self.button_playpause.setIcon(QIcon('icons/play.png'))
            self.learning_thread.learning_algorithm.pause()

        selected_cells = get_selected_cell(self.table)
        if len(selected_cells)==0:
            msg = QMessageBox()
            msg.setWindowTitle('Select Items for Deletion...')
            
            msg.setText("No data selected. Select row(s) to be deleted from the results table, then click the delete icon.")
            
            ok_button = QPushButton('Ok')
            msg.addButton(ok_button, QMessageBox.ButtonRole.AcceptRole)

            # cancel_button = QPushButton('Cancel')
            # msg.addButton(cancel_button, QMessageBox.ButtonRole.RejectRole)

            # # # display the message box and wait for a button to be clicked
            msg.exec()
            

            # # handle the button click
            if msg.clickedButton() == ok_button:
                return

                
        
        del_rows = np.unique([item.row() for item in selected_cells])
        del_rows = [self.results_index[row] for row in del_rows]

        del_df = self.results_data.iloc[del_rows]
        
        

    # Create a QTableWidget
        table = QTableWidget()
        table.setColumnCount(del_df.shape[1])
        table.setRowCount(del_df.shape[0])
        table.setHorizontalHeaderLabels(list(del_df.columns))

        for row in range(del_df.shape[0]):
            for col in range(del_df.shape[1]):
                # item = QTableWidgetItem(f"Row {row} Col {col}")
                item = QTableWidgetItem(str(del_df.iloc[row, col]))
                # if col == 0:
                item.setBackground(QColor(Qt.GlobalColor.red))
                table.setItem(row, col, item)


        # result = msg.exec()
        self.popup = QDialog()
        popup_layout = QVBoxLayout(self.popup)
        popup_layout.addWidget(QLabel("Are you sure you want to delete the following rows?"))
        popup_layout.addWidget(table)
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        popup_layout.addWidget(button_box)
        self.popup.setLayout(popup_layout)
        button_box.accepted.connect(lambda: self.handle_delete(del_rows))
        button_box.rejected.connect(self.handle_cancel)
        self.popup.exec()

        # ask if sure and list cells to delete
    def handle_delete(self,del_rows):
        
        self.results_data.drop(self.results_data.index[del_rows], axis=0, inplace=True)
        overwrite_results(self.session_id,self.results_data)
        self.popup.accept()
        self.update_table_view()
        self.lossWidget.update_saved_plots()
        self.lossWidget.update()

    def handle_cancel(self):
        self.popup.reject()
        print("User cancelled the delete process")


    def update_table_view(self):
        # create a new data frame
        df = load_results(self.session_id)

        # Define a lambda function to extract the last value of each array
        last_value = lambda x: '{:.2f}%'.format(x[-1])

        # Apply the lambda function to the 'Numbers' column and assign the result to a new column
        df['Accuracy'] = df['accuracy'].apply(last_value)

        # Drop the original 'Numbers' column
        df = df.drop('accuracy', axis=1)

        df = df.loc[:, ['user_name', 'Accuracy', 'epochs', 'datetime']]

        sorted_df = df.sort_values(by=['Accuracy','epochs','user_name'],ascending=[True,True,True])
        self.results_index = sorted_df.index
        df = sorted_df
        
        # clear the existing model and set its data to the new data frame
        self.model.clear()
        self.model.setColumnCount(df.shape[1])
        self.model.setRowCount(df.shape[0])
        self.model.setHorizontalHeaderLabels(df.columns)

        for row in range(df.shape[0]):
            for column in range(df.shape[1]):
                item = QStandardItem(str(df.iloc[row, column]))
                self.model.setItem(row, column, item)


    def receive_data(self, serialized_data):
        data = pickle.loads(serialized_data)
        return data

app = QApplication(sys.argv)

window = MainWindow()
display_gui(window,0) # select which monitor to display GUI
app.exec()
