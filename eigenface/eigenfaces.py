import matplotlib.pyplot as plt
import numpy as np

class EigenFaces(object):
    def __init__(self):
        self.mean_face = None
        self.eigen_faces = None

        self.vector_mean_matrix = None
        self.mean_vector = None
        self.eigen_value = None
        self.norm_ui = None
        
        self.weights = None
        self.size = None
        self.percent = 0.9

    # __init__

    def update(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        (self.mean_face, self.eigen_faces), (self.vector_mean_matrix, self.mean_vector, self.eigen_value, self.norm_ui) = calculate_eigen_faces(self.x_train)

        
        self.size    = find_size(self.eigen_value, self.percent)
        self.weights = get_all_weight(self.x_train, self.mean_vector, self.norm_ui, self.size)
    # update
    
    def plot_mean_face(self):
        plot_image(self.mean_face, 'Mean Face')
    # plot_mean_face

    def plot_eigen_faces(self, start = 0, size = [4,4], wspace=1.5, hspace=1.5, fig_size = (10, 10)):
        plot_images(self.eigen_faces, self.y_train, start, size, wspace, hspace, fig_size)
    # plot_mean_face

    def calc_weight(self, image):
        return get_weight(image.flatten(), self.mean_vector, self.norm_ui, self.size)
       
    def predict(self, image):
        weight_vector = self.calc_weight(image)
        (closest_face_id, norm_weight_vector) = distance_classify(weight_vector, self.weights)
        label = self.y_train[closest_face_id]
        return (label, closest_face_id, norm_weight_vector)

    def evaluation(self, x_test, y_test):
        cnt_true  = 0
        cnt_total = len(x_test)
        for cnt in range(len(x_test)):
            (predict_label, predict_closest_face_id, predict_norm_weight_vector) = self.predict(x_test[cnt])
            truth_label = y_test[cnt]
            if predict_label[0] == truth_label[0]:
                cnt_true = cnt_true + 1
        return (cnt_true, cnt_total)
# def

# EigenFaces

def plot_images(images, labels, start = 0, size = [4,4], wspace=2.5, hspace=2.5, fig_size = (10, 10)):
    r, c = size
    fig, axs = plt.subplots(r, c)
    plt.figure(figsize=fig_size, dpi=180)
    fig.set_size_inches(fig_size)
    cnt = 0
    for i in range(r):
        for j in range(c):
            if images.shape[3] == 1:
                axs[i,j].imshow(images[start+cnt, :,:, 0], cmap='gray', aspect='auto')
            else:
                axs[i,j].imshow(images[start+cnt, :,:, :], aspect='auto')
            axs[i,j].axis('off')
            if len(labels.shape)==1:
                axs[i,j].set_title('%s'%(labels[cnt]))
            elif len(labels.shape)==2 and len(labels[cnt]) == 1:
                axs[i,j].set_title('%s'%(labels[cnt, 0]))
            elif len(labels.shape)==2 and len(labels[cnt]) >= 2:
                axs[i,j].set_title('%s (%s)'%(labels[cnt, 0], labels[cnt, 1]))
            cnt += 1
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    plt.show()
# plot_images

def plot_image(image, label):
    if image.shape[2] == 1:        
        plt.imshow(image[:,:,0], cmap='gray')
    else:
        plt.imshow(image[:,:,:])
    if type(label)==str:
        plt.title(label)
    if type(label) is np.ndarray:
        if len(label)==1:
            plt.title('%s'%(label[0]))
        elif len(label)>=2:
            plt.title('%s (%s)'%(label[0], label[1]))
    plt.axis('off')
    plt.show()
# plot_image

def gen_images():
    images = np.empty(shape=(3,4,4))
    images[0,:,:] = np.array([[1,2,3,4],[5,6,7,8],[5,6,7,8],[5,6,7,8]])
    images[1,:,:] = np.array([[2,3,4,5],[6,7,8,9],[5,6,7,8],[5,6,7,8]])
    images[2,:,:] = np.array([[0,2,4,6],[3,5,7,9],[5,6,7,8],[5,6,7,8]])
    return images

def convert_matrix_presentation(images): 
    vector2d = []
    for image in images:
        vector = image.flatten()
        vector2d.append(vector)
    return np.array(vector2d)

def calculate_eigen_vectors(vector_matrix):
    mean_vector = vector_matrix.mean(axis=0)
    vector_mean_matrix = vector_matrix[:,:] - mean_vector
    covariance_matrix = np.matmul(vector_mean_matrix,vector_mean_matrix.T) # vector_matrix: [M x N^2], [N^2 x M]
    u, eigen_value, eigen_vector_vi = np.linalg.svd(covariance_matrix)     # eigen_value: 1 x M, eigen_vector_vi: M x M
    # vector_mean_matrix.T (N^2 x M) x eigen_vector_vi.T (M x 1) = N^2 x 1
    # M eigen vectors with high values
    eigen_vector_ui = np.matmul(vector_mean_matrix.T, eigen_vector_vi[:,:].T).T   
    # normalize eigen vectors
    norms = np.linalg.norm(eigen_vector_ui, axis=1)   # N^2 x 1
    norm_ui = np.divide(eigen_vector_ui.T, norms).T   # 1 x N^2
    return (vector_mean_matrix, mean_vector, eigen_value, norm_ui) # M x N^2, 1 x N^2

def calculate_eigen_faces(images):
    vector_matrix = convert_matrix_presentation(images)
    (vector_mean_matrix, mean_vector, eigen_value, norm_ui) = calculate_eigen_vectors(vector_matrix)
    eigen_faces = norm_ui.reshape(images.shape)
    mean_images = mean_vector.reshape(images.shape[1], images.shape[1], 1)
    return (mean_images, eigen_faces), (vector_mean_matrix, mean_vector, eigen_value, norm_ui)

def get_weight(face_vector, mean_vector, norm_ui, size):
    theta = face_vector - mean_vector
    return np.matmul(norm_ui[:size], theta)

def find_size(eigen_value, percent = 0.9):
    total = eigen_value.sum()
    for i in range(len(eigen_value)):
        size = i + 1
        cur  = eigen_value[:size].sum()
        if cur/float(total)>=percent:
            return size
    return len(eigen_value)

def get_all_weight(images, mean_vector, norm_ui, size):
    w = [get_weight(images[i,:,:].flatten(), mean_vector, norm_ui, size)for i in range(images.shape[0])]
    return w

def distance_classify(w, weights):
    diff = weights - w
    norm_weight = np.linalg.norm(diff, axis=1)
    closest_face_id = np.argmin(norm_weight)
    return (closest_face_id, norm_weight)