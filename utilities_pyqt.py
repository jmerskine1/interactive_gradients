from PyQt6.QtGui import (QScreen, 
                         QPalette, 
                         QColor, 
                         QLinearGradient, 
                         QBrush,
                         QFont,
                         QMouseEvent, 
                         QPainter,
                         QIcon,
                         QPen, 
                         QResizeEvent)

from PyQt6.QtWidgets import (QWidget,
                             QGraphicsOpacityEffect,
                             QLabel,
                             QLayout,
                             QHBoxLayout,
                             QVBoxLayout, 
                             QMainWindow, 
                             QLineEdit,
                             QListWidget,
                             QListWidgetItem,
                             QPushButton,
                             QSplitter)

from PyQt6.QtCore import(pyqtSignal,
                         Qt,
                         QPoint,
                         QPointF,
                         QTimer, 
                         QAbstractTableModel, 
                         QThread, 
                         QObject, 
                         QEvent)

import yaml
import pandas as pd
import numpy as np
import jax.numpy as jnp
import pickle

import datetime
import os.path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap

from utilities import custom_models
from interactive_pipeline import Pipeline



def display_gui(window,monitor_num=None):
    
    if monitor_num:
        monitors = QScreen.virtualSiblings(window.screen())
        monitor = monitors[monitor_num].availableGeometry()
        window.move(monitor.left(), monitor.top())
    
    window.show()

def update_dataset_config(dataset,size):
    if size > 0:
        with open("config.yaml", "r") as yamlfile:
            config = yaml.load(yamlfile, Loader=yaml.FullLoader)
            config[0]['data_params']['dataset'] = dataset
            config[0]['data_params']['size'] = size
            yamlfile.close()

        with open("config.yaml", 'w') as yamlfile:
            data1 = yaml.dump(config, yamlfile)
            yamlfile.close()
    else:
        # disregard the size change
        with open("config.yaml", "r") as yamlfile:
            config = yaml.load(yamlfile, Loader=yaml.FullLoader)
            config[0]['data_params']['dataset'] = dataset
            yamlfile.close()

        with open("config.yaml", 'w') as yamlfile:
            data1 = yaml.dump(config, yamlfile)
            yamlfile.close()

def load_results(session):
    pickle_file = 'interactive_results/' + session + '.pkl'

    # Check if the pickle file exists
    if not os.path.isfile(pickle_file):
        # Create a new DataFrame with user name, datetime, accuracy, and epochs
        df = pd.DataFrame(columns=['user_name', 'accuracy', 'epochs', 'datetime'])
    else:
        # Load the existing DataFrame from the pickle file
        df = pd.read_pickle(pickle_file)

    return df

def store_results(session,user_name,pipeline):
    # Define the name of the pickle file
    pickle_file = 'interactive_results/' + session + '.pkl'

    # Check if the pickle file exists
    if not os.path.isfile(pickle_file):
        # Create a new DataFrame with user name, datetime, accuracy, and epochs
        df = pd.DataFrame(columns=['user_name', 'accuracy', 'epochs', 'datetime'])
    else:
        # Load the existing DataFrame from the pickle file
        df = pd.read_pickle(pickle_file)

    # Add a new row to the DataFrame with the current pipeline info and user name
    current_time = datetime.datetime.now()
    accuracy = pipeline.accs
    epochs = len(accuracy)
    df = df._append({
        'user_name': user_name,
        'accuracy': accuracy,
        'epochs': epochs,
        'datetime': current_time
    }, ignore_index=True)

    # Store the DataFrame as a pickle file
    df.to_pickle(pickle_file)

def overwrite_results(session,df):
    # Define the name of the pickle file
    pickle_file = 'interactive_results/' + session + '.pkl'
    df.to_pickle(pickle_file)



def get_selected_cell(table):
    return table.selectedIndexes()
    

class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=10, height=10, dpi=100):
        
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.subplots_adjust(left=0.1,bottom=0.1,right=0.95,top=0.95)
        self.axes = self.fig.add_subplot(111)

        super(MplCanvas, self).__init__(self.fig)

class Color(QWidget):

    def __init__(self, color):
        super(Color, self).__init__()
        self.setAutoFillBackground(True)

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(color))
        self.setPalette(palette)

class InProgress(QWidget):

    def __init__(self):
        super(InProgress, self).__init__()
        
        # Set initial gradient parameters
        self.gradient_start = QColor('red')
        self.gradient_end = QColor('blue')
        self.centertext = QLabel('In Progress',self)
        self.centertext.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.centertext.setGeometry(self.rect())
        font = QFont()
        font.setFamily("Helvetica")
        font.setPointSize(int(self.geometry().height()/5))
        self.centertext.setFont(font)

    def paintEvent(self, event):
        # Create linear gradient
        geometry = self.geometry()
        gradient = QLinearGradient(0,0,geometry.width(),geometry.height())
        gradient.setColorAt(0, self.gradient_start)
        gradient.setColorAt(1, self.gradient_end)
        
        # Paint background with gradient
        painter = QPainter(self)
        painter.fillRect(self.rect(), gradient)
        
    def resizeEvent(self, event: QResizeEvent):
        # Update gradient parameters when widget is resized
        self.gradient_start = QColor('red')
        self.gradient_end = QColor('blue')
        self.centertext.setGeometry(self.rect())
        font = self.centertext.font()
        font.setPointSize(int(self.geometry().height()/5))
        self.centertext.setFont(font)

        self.update()
        QMainWindow.resizeEvent(self, event)


class LossMPL(QWidget):
    def __init__(self, pipeline, session_id, *args, **kwargs):
        super(QWidget, self).__init__(*args, **kwargs)
        self.session_id=session_id
        self.canvas = MplCanvas(self, width=10, height=8, dpi=100)

        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.setLayout(layout)
        

        
        self.xdata = []
        self.ydata = []
        self.xmax = 0
        self.saved_xmax = 0

        # We need to store a reference to the plotted line
        # somewhere, so we can apply the new data to it.
        self._plot_ref = None
        self.legend = self.canvas.axes.legend()
    
        
        self.saved_plots =[]
        self.reset(pipeline)
        self.show()


    def update_plot(self, pipeline):

        if self._plot_ref is None:
            self.xdata = [0]
            self.ydata = [0]

            # First time we have no plot reference, so do a normal plot.
            # .plot returns a list of line <reference>s, as we're
            # only getting one we can take the first element.
            
            self.plot_refs = self.canvas.axes.plot(self.xdata, self.ydata,label='Current')
            self.update_saved_plots(session_id=self.session_id)
            
            self.canvas.axes.tick_params(axis='both', which='major',pad=0)
            
            self.canvas.axes.set_ylim([0,105])
            
            self._plot_ref = self.plot_refs[0]
        else:
            self.xdata = list(range(0,len(pipeline.accs)))
            self.ydata = pipeline.accs

            self._plot_ref.set_xdata(self.xdata)
            self._plot_ref.set_ydata(self.ydata)

            if len(self.xdata) > self.saved_xmax:
                # self.old_max = self.x_max
                self.xmax = max(len(self.xdata),self.xmax)
                self.canvas.axes.set_xlim([0,self.xmax])
            else:
                self.canvas.axes.set_xlim([0,self.saved_xmax])

        # Trigger the canvas to update and redraw.
        self.canvas.draw()
    
    def reset(self, pipeline):

        self.xdata = []
        self.ydata = []

        pipeline.__init__()
        self.update_saved_plots(self.session_id)
        self.update_plot(pipeline)

    def update_saved_plots(self,session_id):

        for loss_plot in self.saved_plots:
            # l = loss_plot[0]
            loss_plot.remove()

        self.legend.remove()
        
        self.saved_plots =[]
        saved_data = load_results(session=session_id)

        for index, row in saved_data.iterrows():
            x = range(len(row['accuracy']))
            self.saved_xmax = max(x[-1],self.saved_xmax)
            y = row['accuracy']
            label = row['user_name']
            self.saved_plots = np.append(self.saved_plots, self.canvas.axes.plot(x,y,label=label,zorder=1))
            
        self.canvas.axes.set_xlim([0,self.saved_xmax+1e-5])
        self.canvas.axes.set_xlabel("Epochs",fontsize=10)
        self.canvas.axes.set_ylabel("Accuracy",fontsize=10)
        self.legend = self.canvas.axes.legend()
        self.canvas.draw()

    
    def resizeEvent(self, event: QResizeEvent):
        # Maintain a square aspect ratio by setting width and height to the minimum of the two
        self.canvas.fig.tight_layout()
        size = min(self.width(), self.height())
        self.resize(int(1.3*size), size)     
           
        


class ScatterMPL(QWidget):
    selectionSignal = pyqtSignal(list)
    selectNoneSignal = pyqtSignal(int)
    def __init__(self, pipeline, *args, **kwargs):
        super(QWidget, self).__init__(*args, **kwargs)

        self.canvas = MplCanvas(self, width=10, height=10, dpi=100)
        self.canvas.mpl_connect('button_press_event',self.mouseClickEvent)
        layout = QVBoxLayout()
        
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
        self.setMouseTracking(True)

        # We need to store a reference to the plotted line
        # somewhere, so we can apply the new data to it.
        # self._plot_ref = None
        self._cont_ref = None
        self.regenerate_data(pipeline)
        
        self.selected_point = None
        self.circle=None

        self.show()

    def regenerate_data(self, pipeline):
        pipeline.__init__()
                
    
    def update_plot(self, pipeline):
        
                # get data out of pipeline
        self.hyperparams = pipeline.hyperparams
        self.X = pipeline.train_dataset.data.X
        self.y = pipeline.train_dataset.data.Y
        self.K = pipeline.train_dataset.data.K
        self.state = pipeline.state
        self.selected_point = None
        self.sendNone()
        self.circle = None
        

        x_min, x_max =self.X[:, 0].min() - .5, self.X[:, 0].max() + .5
        y_min, y_max = self.X[:, 1].min() - .5, self.X[:, 1].max() + .5
        lims = [[x_min, x_max], [y_min, y_max]]
        self.xx, self.yy = np.meshgrid(np.arange(lims[0][0], lims[0][1], 0.01),
                                np.arange(lims[1][0], lims[1][1], 0.01))
        points = np.stack([self.xx.ravel(), self.yy.ravel()]).T
    
        model = custom_models[self.hyperparams['model']](*self.hyperparams['model_size'])
        self.Z = model.apply({'params': self.state.params}, points)

        self.cm = plt.cm.RdBu
        self.cm_bright = ListedColormap(['#FF0000', '#0000FF'])

        self.canvas.axes.cla()
        self.scatter = self.canvas.axes.scatter(self.X[:, 0], self.X[:, 1], c=self.y,
                                                  cmap=self.cm_bright,edgecolors='k', zorder=10)
        
        self.canvas.axes.set_xlim(lims[0])
        self.canvas.axes.set_ylim(lims[1])
        self.canvas.axes.set_xlabel("Feature 0",fontsize=10)
        self.canvas.axes.set_ylabel("Feature 1",fontsize=10)
        self.canvas.axes.tick_params(axis='both', which='major', size = 2,labelsize=5,pad=0)
        
        for x,ks in list(zip(self.X,self.K['vector'])):
            if len(ks) == 0:
                continue
            else:
                for k in ks:
                    U = np.array([k[0]])
                    V = np.array([k[1]])
                    # Add check to prevent divide by zero warning
                    divisor = np.sqrt(U**2 + V**2)
                    divisor[divisor == 0] = 1  # set divisor to 1 where it is zero
                    U /= divisor
                    V /= divisor
                    self.canvas.axes.quiver(x[0],x[1],U,V,color='gold',
                                            transform=self.canvas.axes.transData,zorder = 9)

        # Check for existing contour plot, and plot a new one 
        if self._cont_ref is None:
            cont_refs = self.canvas.axes.contourf(self.xx, self.yy, 
                                                  self.Z.reshape(self.xx.shape), cmap=self.cm, alpha=.8,
                                                  zorder = 1)
            self._cont_ref = cont_refs
        else:
            for coll in self._cont_ref.collections:
                coll.remove()
            
            self._cont_ref = self.canvas.axes.contourf(self.xx, self.yy, 
                                                  self.Z.reshape(self.xx.shape), cmap=self.cm, alpha=.8,
                                                  zorder = 1)

        # Trigger the canvas to update and redraw.
        self.canvas.draw()
    
    def circleDraw(self):
            # calculate the x and y limits of the plot
        x_limits = self.canvas.axes.get_xlim()
        y_limits = self.canvas.axes.get_ylim()

        if self.circle is not None:
            self.circle.remove()
        # calculate the radius of the circle as a function of the x and y limits
        radius = 0.015 * (x_limits[1] - x_limits[0] + y_limits[1] - y_limits[0])
        # radius = 0.1
        self.circle = Circle((self.selected_point[0],self.selected_point[1]),radius,
                             transform=self.canvas.axes.transData,
                         facecolor='none', edgecolor = 'lightslategray', 
                          linestyle='--',zorder=20)
        try:
            self.q.remove()
        except:
            pass
        
        self.circle.set_linewidth(2)
        self.canvas.axes.add_artist(self.circle)
        self.canvas.draw()
        


    def mouseClickEvent(self, event):
           # get the x and y coordinates of the mouse click
        x, y = event.x, event.y

        # get the x and y positions of the plotted points
        offsets = self.scatter.get_offsets()
        trans = self.canvas.axes.transData
        pixel_coords = trans.transform(offsets)

        # calculate the distance between the mouse click position and each plotted point
        distances = np.sqrt((pixel_coords[:, 0] - x) ** 2 + (pixel_coords[:, 1] - y) ** 2)

        # select the point with the smallest distance
        nearest_point = np.argmin(distances)
        self.selected_point = [offsets[nearest_point, 0], offsets[nearest_point, 1]]

        self.circleDraw()
        self.sendSignal()
    
    def sendSignal(self):
        self.selectionSignal.emit(self.selected_point)
    
    def sendNone(self):
        self.selectNoneSignal.emit(2)

    def live_vectors(self,value1, value2):
        try:
            self.q.remove()
        except:
            pass
        U = np.array([value1])
        V = np.array([value2])
        # Add check to prevent divide by zero warning
        divisor = np.sqrt(U**2 + V**2)
        divisor[divisor == 0] = 1  # set divisor to 1 where it is zero
        U /= divisor
        V /= divisor
        self.q = self.canvas.axes.quiver(self.selected_point[0],self.selected_point[1],U,V,
                                         color='forestgreen',transform=self.canvas.axes.transData)
        self.canvas.draw()
    
    def resizeEvent(self, event: QResizeEvent):
        # Maintain a square aspect ratio by setting width and height to the minimum of the two
        
        size = min(self.width(), self.height())
        self.resize(size, size)      
        self.canvas.fig.tight_layout()



class TableModel(QAbstractTableModel):

    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if role == Qt.ItemDataRole.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            return str(value)

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]

    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._data.columns[section])

            if orientation == Qt.Orientation.Vertical:
                return str(self._data.index[section])

class LearningAlgorithm(QObject):
    # stateUpdate = pyqtSignal(FrozenDict)
    stateUpdate = pyqtSignal(Pipeline)
    # stateUpdate = pyqtSignal(TrainState)
    

    def __init__(self,pipeline, parent=None):
        super().__init__(parent)
        self._running = False
        self.pipeline = pipeline

    def start(self):
        
        self._running = True
        # self.pipeline = pipeline
        while self._running:
            # Run the learning algorithm
            self.pipeline.train_one_epoch()
            self.stateUpdate.emit(self.pipeline)

            QThread.msleep(100)

    def send_data(self):
        serialized_data = pickle.dumps(self.pipeline)
        self.stateUpdate.emit(serialized_data)       
            
        
    def pause(self):
        self._running = False
        # return self.pipeline
    
    def quit(self):
        self._running = False
        self.quit()
    # def update(self,pipeline):
    #     self.pipeline = 

    def is_running(self):
        return self._running
               
class LearningThread(QThread):
    def __init__(self, pipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.learning_algorithm = LearningAlgorithm(self.pipeline)
        
    def run(self):
        self.learning_algorithm.start()

class KnowledgeList(QWidget):
    knowledgestack_index = pyqtSignal(int)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.sendNone()
        self.list_widget = QListWidget(self)
    
    def get_knowledge_list(self,pipeline,scatter):
        self.get_index(pipeline,scatter)
        len_knowledge = len(pipeline.train_dataset.data.K['vector'][self.index])
        if len_knowledge==0:
            self.widget = QLabel('There is no knowledge available. Hit the "'"+"'" button to add knowledge')
        else:
            vectors = []
            magnitudes = []
            self.magnitudes = [1]*len_knowledge
            # gather up all knowledge
            for i in range(len_knowledge):
                self.vectors = np.append(self.vectors,pipeline.train_dataset.data.K['vector'][self.index][i])
                item_widget = KnowledgeListItem(f'Entry {i}',self.list_widget)
                item = QListWidgetItem(self.list_widget)
                item.setSizeHint(item_widget.sizeHint())
                self.list_widget.addItem(item)
                self.list_widget.setItemWidget(item, item_widget)
            #plot vectors

            self.widget = self.list_widget

        
    def sendNone(self):
        self.knowledgestack_index.emit(2)
        

    def get_index(self,pipeline,scatter):
        selected_point = scatter.selected_point
        if selected_point == None:
            self.sendNone()


        self.index = [i for i, coor in enumerate(pipeline.train_dataset.data.X) if coor[0]==selected_point[0] 
                 and coor[1] == selected_point[1]]
        

class KnowledgeListItem(QWidget):
    def __init__(self, text, list_widget):
        super().__init__()
        self.list_widget = list_widget
        self.text = text

        # Create label to display text
        self.label = QLabel(self.text)
        
        # Create button to delete item
        self.delete_button = QPushButton()
        self.delete_button.setIcon(QIcon('icons/clear.png'))
        self.delete_button.clicked.connect(self.delete_item)

        # Create layout and add label and delete button
        layout = QHBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.delete_button)
        self.setLayout(layout)
    
    def delete_item(self):
        # Get index of this item in the list widget
        index = self.list_widget.indexFromItem(self.list_widget.item(self.list_widget.row(self))).row()

        # Remove item from list widget
        self.list_widget.takeItem(index)

