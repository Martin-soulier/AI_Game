from tkinter import ALL, HIDDEN
import numpy as np
import pygame
import time 
import random

pygame.init()


TAILLE_GRILLE_JEU = 7
TAILLE_IMAGE = 50
RECOMPENSE_NEGATIVE = -1
RECOMPENSE_POSITIVE = 1
DEPLACEMENT_TABLEAU = [(0,1), (1,0), (0,-1), (-1,0)]
NOMBRE_DONNEES_ETAT = 12
TAILLE_MEMOIRE_MOINS_GRANDE = 32
LEARNING_RATE = 0.001
NBR_CHANGEMENT_EPS = 750
NBR_AFFICHAGE_PARTIE = 20
COULEUR_SCORE = (255, 255, 255)
HIDDEN_LAYERS = [8,8]

#mettre la page pygame

ecran = pygame.display.set_mode((TAILLE_IMAGE*TAILLE_GRILLE_JEU, TAILLE_IMAGE*TAILLE_GRILLE_JEU))
police = pygame.font.SysFont('monospace', TAILLE_IMAGE)
# télécharger toutes les images grâce a pygame

LOAD_TETE_SERPENT = pygame.image.load("tete_serpent.png").convert()
TETE_SERPENT = pygame.transform.scale(LOAD_TETE_SERPENT, (TAILLE_IMAGE, TAILLE_IMAGE))

LOAD_CORPS_SERPENT =  pygame.image.load("corps_serpent.png").convert()
CORPS_SERPENT = pygame.transform.scale(LOAD_CORPS_SERPENT, (TAILLE_IMAGE, TAILLE_IMAGE))

LOAD_MUR = pygame.image.load("murs.png").convert()
MUR = pygame.transform.scale(LOAD_MUR, (TAILLE_IMAGE, TAILLE_IMAGE))

LOAD_CASE_VIDE = pygame.image.load("case_vide.png").convert()
CASE_VIDE = pygame.transform.scale(LOAD_CASE_VIDE, (TAILLE_IMAGE, TAILLE_IMAGE))

LOAD_POMME = pygame.image.load("pomme.png").convert()
POMME = pygame.transform.scale(LOAD_POMME, (TAILLE_IMAGE, TAILLE_IMAGE))

ALL_IMAGES = [CASE_VIDE, MUR, TETE_SERPENT, CORPS_SERPENT, POMME]

pygame.quit()

class Game():

    def __init__(self):
        self.grille_jeu, self.position_serpent, self.direction_serpent, self.position_pomme, self.direction_pomme_depart = self.reset_grille()
        self.taille = 2
        self.score = 0
        self.mouvement = 0

    def reset_grille(self):

        '''Ne prend rien en paramètre
        Initialise le tableau de jeu' avec les cases vides (0), les murs (1), la tête du serpent (2), le corps
        du serpent (3) et la pomme (4)
        renvoie la grille de jeu qui est un tableau numpy à double entrées
        renvoie la position du serpent qui est un tuple de deux case avec ses coordonnées 
        renvoie la direction du serpent qui est un nombre entre 0 et 3, 0 indique la droite puis on tourne 
        dans le sens des aiguilles d'une montre
        renvoie la position de la pomme qui est un tuple de deux cases avec ses coordonnées'''
        
        # créer la grille 
        grille_jeu = np.zeros((TAILLE_GRILLE_JEU, TAILLE_GRILLE_JEU))
        grille_jeu[0] = np.ones((TAILLE_GRILLE_JEU))
        grille_jeu[TAILLE_GRILLE_JEU-1] = 1

        for i in range(1, TAILLE_GRILLE_JEU-1):
            grille_jeu[i][0] = 1
            grille_jeu[i][TAILLE_GRILLE_JEU-1] = 1
        

        #place le serpent
        x = random.randint(3, TAILLE_GRILLE_JEU-4)
        y = random.randint(3, TAILLE_GRILLE_JEU-4)
        direct_serpent = random.randint(0,3)
        all_position_serpent = [[(y, x+2), (y,x+1), (y,x)], [(y+2, x), (y+1,x), (y,x)],[(y, x-2), (y,x-1), (y,x)], [(y-2, x), (y-1,x), (y,x)]]
        for i in range(2):
            grille_jeu[all_position_serpent[direct_serpent][i][0]][all_position_serpent[direct_serpent][i][1]] = 3
        grille_jeu[all_position_serpent[direct_serpent][2][0]][all_position_serpent[direct_serpent][2][1]] = 2
        position_serpent = all_position_serpent[direct_serpent]

        #place la pomme
        direct_pomme = random.randint(0,3)
        while direct_pomme == direct_serpent:
            direct_pomme = random.randint(0,3)
        
        all_position_pomme = [(y, x+1),(y+1, x), (y, x-1), (y-1, x)]
        grille_jeu[all_position_pomme[direct_pomme][0]][all_position_pomme[direct_pomme][1]] = 4
        position_pomme = all_position_pomme[direct_pomme]
        
        #a voir si c vrmt utile 
        direction_op = [2,3,0,1]
        direction_depart = direction_op[direct_serpent]

        return grille_jeu, position_serpent, direction_depart, position_pomme, direct_pomme


    def affichage(self, ecran, police):
        '''Ne prend rien en parametre
        permet l'affichage graphique du tableau 'grille_jeu' grâce ) pygame
        Ne renvoie rien '''
        for i in range(TAILLE_GRILLE_JEU):
            for j in range(TAILLE_GRILLE_JEU):
                ecran.blit(ALL_IMAGES[int(self.grille_jeu[i][j])], (j*TAILLE_IMAGE, i*TAILLE_IMAGE))
                affichage_score = police.render(str(self.score), 1, COULEUR_SCORE)
        ecran.blit(affichage_score, (TAILLE_IMAGE*TAILLE_GRILLE_JEU//2, 0))
        pygame.display.flip()
        return None


    def placer_pomme(self):
        '''Ne prend rien en paramètre
        modifie le tableau 'grille_jeu' afin de placer une pomme aléatoirement sur la carte
        renvoie la position de la pomme qui est un tuple de deux cases avec ses coordonnées '''
        placement = True
        while placement:
            x = random.randint(1, TAILLE_GRILLE_JEU-2)
            y = random.randint(1, TAILLE_GRILLE_JEU-2)
            if self.grille_jeu[y][x] == 0:
                self.grille_jeu[y][x] = 4
                placement = False
        position_pomme = (y, x)
        return position_pomme


    def mouvement_possible(self):
        '''Ne prend rien en paramètre 
        Renvoie un tableau avec les 4 valeurs autour du serpent
        les indices veulent dire : 0 vers la droite, 1 vers le bas, 2 vers la gauche, 3 vers le haut '''
        autour_serpent = [0,0,0,0]
        autour_serpent[0] = self.grille_jeu[self.position_serpent[self.taille][0]][self.position_serpent[self.taille][1]+1]
        autour_serpent[1] = self.grille_jeu[self.position_serpent[self.taille][0]+1][self.position_serpent[self.taille][1]]
        autour_serpent[2] = self.grille_jeu[self.position_serpent[self.taille][0]][self.position_serpent[self.taille][1]-1]
        autour_serpent[3] = self.grille_jeu[self.position_serpent[self.taille][0]-1][self.position_serpent[self.taille][1]]
        return autour_serpent
    

    def deplacement(self):
        '''Ne prend rien en paramètre
        Modifie le tableau 'grille_jeu' avec le déplacement effectuer par le joueur
        Renvoie la récompense recu par le serpent qui est un nombre
        Renvoie True si le jeu continuer ou False si la partie est fini'''
        
        if self.mouvement == 0:
            self.direction_serpent = self.direction_pomme_depart
            
        self.mouvement += 1

        autour_serpent = self.mouvement_possible()
        if autour_serpent[self.direction_serpent] % 2 == 1:
            return RECOMPENSE_NEGATIVE, False

        manger_pomme = False
        if autour_serpent[self.direction_serpent] == 4:
            manger_pomme = True

        new_x = self.position_serpent[self.taille][0] + DEPLACEMENT_TABLEAU[self.direction_serpent][0]
        new_y = self.position_serpent[self.taille][1] + DEPLACEMENT_TABLEAU[self.direction_serpent][1]

        self.grille_jeu[new_x][new_y] = 2
        self.position_serpent.append((new_x, new_y))
        self.grille_jeu[self.position_serpent[self.taille][0]][self.position_serpent[self.taille][1]] = 3

        if manger_pomme:
            self.taille += 1
            self.score += 1 
            self.position_pomme = self.placer_pomme()
            return RECOMPENSE_POSITIVE, True
        
        self.grille_jeu[self.position_serpent[0][0]][self.position_serpent[0][1]] = 0
        del self.position_serpent[0]
        
        return 0, True

        
    def faire_etat(self):
        '''Ne prend rien en paramètre
        Renvoie l'etat du serpent qui est un tableau de dimension (1,n)'''

        etat = np.zeros((NOMBRE_DONNEES_ETAT))
        compt = 0
        
        for i in range(-1, 2):
            x = i+self.position_serpent[self.taille][1]

            for j in range(-1, 2):
                distance = 0
                y = j + self.position_serpent[self.taille][0]

                if i != 0 or j != 0:
                    add_x = 0
                    add_y = 0
                    while int(self.grille_jeu[x+add_x][y+add_y]) != 1 and int(self.grille_jeu[x+add_x][y+add_y]) != 3:
                        add_x += i
                        add_y += j
                        distance += 1
                    etat[compt] = distance
                    compt += 1
        
                   
        # voir de quelle côté le serpent doit aller pour aller vers la pomme, 1 il doit y aller 0 sinon
        direction_pomme = (self.position_serpent[self.taille][0] - self.position_pomme[0], self.position_serpent[self.taille][1] - self.position_pomme[1])
        coup_vers_pomme = [0,0,0,0]

        if direction_pomme[0] > 0:
            coup_vers_pomme[3] = direction_pomme[0]
        if direction_pomme[0] < 0:
            coup_vers_pomme[1] = direction_pomme[0]*-1
        
        if direction_pomme[1] > 0:
            coup_vers_pomme[2] = direction_pomme[1]
        if direction_pomme[1] < 0:
            coup_vers_pomme[0] = direction_pomme[1]*-1
        
        for val in coup_vers_pomme:
            etat[compt] = val
            compt += 1

        return np.array([etat])

        


class Reseau_neurones():

    def __init__(self):
        self.parametres = self.initialisation()

    def initialisation(self):
        '''Ne prend rien en paramètre
        renvoie un dictionnaire qui contient les poids des différents couches du réseau de neurone'''
        C = len(HIDDEN_LAYERS)+2
        dimension = [0]*C
        dimension[0] = NOMBRE_DONNEES_ETAT
        for i in range(1, C-1):
            dimension[i] = HIDDEN_LAYERS[i-1]
        dimension[C-1] = 4
        parametres = {}
        
        for c in range(1,C):
            parametres['W' + str(c)] = np.random.uniform(-1, 1, size=(dimension[c-1], dimension[c]))
            parametres['b' + str(c)] = np.random.uniform(-1, 1, size=(1, dimension[c]))
        

        return parametres
    
    def relu(self, tab):
        '''Prend en parametre un tableau de dimension (1,n) avec des nombres
        Elle modifie ce tableau grâce a la fonction relu
        Renvoie ce tableau'''
        for i in range(len(tab)):
            for j in range(len(tab[0])):
                if tab[i][j] < 0:
                    tab[i][j] *= 0.1
        return tab

    def forward_propagation(self, etat):
        '''Prend en paramètre un tableau de dimension (1, NOMBRE_DONNEES_ETAT)
        Fait l'étape de la forward propagation dans mon réseau
        Renvoie les activation de chaque couche qui est un dictionnaires contenant des tableaux 
        Renvoie les activation de chaque couche avant d'avoir utiliser la fonction relu qui est un dico contenant
        des tableaux'''

        C = len(self.parametres) // 2
        activation = {'A0' : etat}
        Z = {}
        for c in range(1, C+1):
            Z[c] = (activation['A' + str(c-1)].dot(self.parametres['W' + str(c)])) + self.parametres['b' + str(c)]
            if c != C:
                A = self.relu(np.copy(Z[c]))
            else:
                A = np.copy(Z[c])
            activation['A' + str(c)] = A
        return activation, Z
    
    def derive_relu(self, tab):
        '''Prend en parametre un tableau de dimension (1,n) avec des nombres
        Elle modifie le tableau en faisant la dérivé de la fonction relu (1 si x > 0 et 0 si x < 0)
        Ne renvoie rien'''
        for i in range(len(tab)):
            for j in range(len(tab[0])):
                if tab[i][j] > 0:
                    tab[i][j] = 1
                else:
                    tab[i][j] = 0.1
        return None

    def back_propagation(self, y, activation, Z):
        '''Prend en parametre un tableau qui est la cible du réseau de neurones et les dico activation et Z 
        obtenu grâce a la fonction forward_propagation
        Elle calcul les gradients pour mes différents poids 
        Renvoie tous les gradients dans un dictionnaires contenant des tableaux'''
        
        C = len(self.parametres) // 2
        dZ = 2*(activation['A' + str(C)]-y) 
        gradients = {}
        
        for c in reversed(range(1, C+1)):
            gradients['dW' + str(c)] = (activation['A' + str(c-1)].T).dot(dZ)
            gradients['db' + str(c)] = np.sum(dZ, axis=0, keepdims=True)
            
            if c > 1:
                self.derive_relu(Z[c-1])
                dZ = dZ.dot((self.parametres['W'+str(c)].T)) * Z[c-1]
                
        return gradients
    
    def update(self, gradients, learning_rate):
        '''Prend en parametre le dico gradients obtenu grâce a la back_propagation et un nombre compris entre 0 et 1
        modifie les paramètre du réseau de neurones 
        Ne renvoie rien'''

        C = len(self.parametres) // 2
        for c in range(1, C+1):
            self.parametres['W' + str(c)] = self.parametres['W' + str(c)] - learning_rate * gradients['dW' + str(c)]
            self.parametres['b' + str(c)] = self.parametres['b' + str(c)] - learning_rate * gradients['db' + str(c)]
    
        return None

    def entrainer_reseau(self, etat, target):
        '''Prend en parametre un tableau qui correspond a l'etat du serpent de taille (1, n) et une target qui
        est ce que le réseau de neurone est censé obtenir qui est un tableau de taille (1,n)
        Ne renvoie rien'''
        activations, Z = self.forward_propagation(etat)
        gradients = self.back_propagation(target, activations, Z)
        self.update(gradients, LEARNING_RATE)
        return None


class IA(Game):

    def __init__(self):
        self.memoire = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.min_epsilon = 0.1
        self.mult_epsilon = 0.95
        self.reseau_neurones = Reseau_neurones()
    
    def arg_max(self, tab, val_pos):
        '''Prend en parametre un tableau de dimension (1, n) 
        renvoie l'argument de la plus grande valeur de tab[0]'''
        maxi = tab[0][val_pos[0]]
        arg = val_pos[0]
        for val in val_pos:
            if tab[0][val] > maxi:
                maxi = tab[0][val]
                arg = val
        return arg

    def val_max(self, tab):
        '''Prend en parametre un tableau de taille (1,n)
        renvoie la plus grande valeur de tab[0]'''
        maxi = tab[0][0]
        for val in tab[0]:
            if val > maxi:
                maxi = val
        return maxi

    def choix_action(self, etat, ancienne_direction):
        '''Ne prend rien en paramètre
        Renvoie un nombre entre 0 et 3 qui définit le mouvement que veut faire l'IA'''

        direction_oppose = [2,3,0,1]
        if random.random() <= self.epsilon:
            action = random.randint(0,3)
            while action == direction_oppose[ancienne_direction]:
                action = random.randint(0,3)
            return action
        
        val_pos = [0,1,2,3]
        del val_pos[direction_oppose[ancienne_direction]]
        val_action = self.reseau_neurones.forward_propagation(etat)[0]['A' + str(len(HIDDEN_LAYERS)+1)]
        return self.arg_max(val_action, val_pos) 
        
    def entrainement(self):
        '''Ne prend rien en paramètre
        Va entrainer le réseau de neurone grâce a la descente de gradient
        Ne renvoie rien '''

        memoire_moins_grande = random.sample(self.memoire, TAILLE_MEMOIRE_MOINS_GRANDE)
        data = np.array([[0. for i in range(NOMBRE_DONNEES_ETAT)] for i in range(TAILLE_MEMOIRE_MOINS_GRANDE)])
        y = np.array([[0. for i in range(4)] for i in range(TAILLE_MEMOIRE_MOINS_GRANDE)])
        compt = 0

        for s, a, r, stp1, continuer in memoire_moins_grande:
            
            data[compt] = s
            target = r
            if continuer and r != RECOMPENSE_POSITIVE:
                predict_stp1 = self.reseau_neurones.forward_propagation(stp1)[0]['A' + str(len(HIDDEN_LAYERS)+1)]
                target = r + (self.gamma * self.val_max(predict_stp1))

            target_f = self.reseau_neurones.forward_propagation(s)[0]['A' + str(len(HIDDEN_LAYERS)+1)]
            target_f[0][a] = target
            y[compt] = np.copy(target_f)
            compt += 1
        
        self.reseau_neurones.entrainer_reseau(data, y)

        return None
            


def entrainer_IA(nbr_episode):
    '''Prend en paramètre un entier qui correspond au nombre de parties que fera l'IA
    Apprend l'IA à jouer à snake en affichants parfois les parties de l'IA 
    renvoie et affiche les meilleurs paramètres trouvés'''

    agent = IA()
    jeu = Game()

    compt = 0

    for i in range(1, nbr_episode+1):

        jeu.__init__()
        s = jeu.faire_etat()
        continuer = True

        while continuer:

            jeu.direction_serpent = agent.choix_action(s, jeu.direction_serpent)
            r, continuer = jeu.deplacement()
            stp1 = jeu.faire_etat()
            agent.memoire.append((s, jeu.direction_serpent, r, stp1, continuer))
            if compt > 5000:
                del agent.memoire[0]

            s = stp1
            compt += 1

        if i % NBR_CHANGEMENT_EPS == 0:
            if agent.epsilon > agent.min_epsilon:
                agent.epsilon *= agent.mult_epsilon

        if i%100 == 0:
            print('episode {}/{}, score {}, mouvement {}, epsilon {}'.format(i, nbr_episode, jeu.score, jeu.mouvement, agent.epsilon))
            
        if compt > TAILLE_MEMOIRE_MOINS_GRANDE:
            agent.entrainement()

        if i % 1000 == 0:
            print(agent.reseau_neurones.forward_propagation(s)[0]['A' + str(len(HIDDEN_LAYERS)+1)])
    print('entrainement terminé')
    print(agent.reseau_neurones.parametres)

    return agent.reseau_neurones.parametres


def faire_partie_humain():
    '''Ne prend rien en paramètre
    Permet à l'utilisateur de faire une partie grâce aux flèches pour tourner
    Ne renvoie rien '''
    pygame.init()
    ecran = pygame.display.set_mode((TAILLE_IMAGE*TAILLE_GRILLE_JEU, TAILLE_IMAGE*TAILLE_GRILLE_JEU))
    police = pygame.font.SysFont('monospace', TAILLE_IMAGE)
    jeu = Game()
    jeu.affichage(ecran, police)

    #attendre que le joueur clique sur un bouton avant de commencer à jouer 
    continuer = False
    while not continuer:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                continuer = True
            if event.type == pygame.QUIT:
                pygame.quit()
        pygame.display.flip()

    
    while continuer:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    jeu.direction_serpent = 0
                elif event.key == pygame.K_DOWN:
                    jeu.direction_serpent = 1
                elif event.key == pygame.K_LEFT:
                    jeu.direction_serpent = 2
                elif event.key == pygame.K_UP:
                    jeu.direction_serpent = 3
            elif event.type == pygame.QUIT:
                pygame.quit()
        r, continuer = jeu.deplacement()

        jeu.affichage(ecran, police)
        time.sleep(0.1)

    pygame.quit()
    return None


def faire_jouer_IA(parametres):
    '''Prend en parametre des paramètre pour un réseau de neurone
    affiche la partie de l'IA avec ces paramètres
    Ne renvoie rien'''
    print()
    pygame.init()
    ecran = pygame.display.set_mode((TAILLE_IMAGE*TAILLE_GRILLE_JEU, TAILLE_IMAGE*TAILLE_GRILLE_JEU))
    police = pygame.font.SysFont('monospace', TAILLE_IMAGE)
    
    agent = IA()
    agent.reseau_neurones.parametres = parametres 
    jeu = Game()
    agent.show = True
    agent.epsilon = 0
    
    C = len(parametres)//2
    HIDDEN_LAYERS = [0]*(C-1)
    for i in range(1,C):
      HIDDEN_LAYERS[i-1] = len(parametres['b' + str(i)][0])
    

    continuer = False
    while not continuer:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                continuer = True
            if event.type == pygame.QUIT:
                pygame.quit()
        pygame.display.flip()
    
    while continuer:
        s = jeu.faire_etat()
        jeu.direction_serpent = agent.choix_action(s, jeu.direction_serpent)
        r, continuer = jeu.deplacement()

        jeu.affichage(ecran, police)

        time.sleep(0.3)

    return None


best_para = {'W1': np.array([[ 0.12410548, -0.45358639, -0.92907047, -1.30329589,  0.37395795,
        -0.95822587,  0.66144353,  0.42850193, -1.09153446, -1.12405702,
         0.14063967, -0.36128838, -0.15060432, -0.43853365, -0.80657844,
        -0.84961363],
       [-0.52413042, -0.27122133,  0.60446702,  0.24362307,  0.25223839,
        -0.75717117, -0.27417165, -1.21372536, -0.08239879, -0.58822332,
         0.55119161, -0.93103928, -0.03784209, -0.8263766 , -0.19963949,
         0.16765296],
       [-1.07096878,  0.09044753,  0.7111879 ,  0.07471093, -1.59415667,
        -0.55557992, -0.98723088,  0.17502305, -1.23158762, -1.35234936,
         0.0534849 , -0.58645736, -0.18471272, -0.03143513, -0.66887391,
         0.06364848],
       [-0.59611538, -0.21451039, -0.30559623, -0.7451312 , -0.65238655,
        -0.57828112,  0.56987149, -0.9947719 , -0.18506726,  0.18058836,
         0.65512871, -1.72109536, -0.67134109, -0.60754983, -0.67076094,
        -0.78857684],
       [ 0.90793711, -0.10831433,  0.85187714,  0.13922916, -0.72235406,
        -1.20415269,  0.27159017, -0.89784586,  0.40195272,  0.21760266,
         0.28463319, -0.40638475, -0.86690667, -0.07701705, -0.28487244,
        -1.08113709],
       [ 0.76929455, -0.5138399 , -0.29530406, -0.93862553, -0.42487389,
         0.40233867, -0.31132876, -0.52890194, -1.12299993, -0.04009911,
        -0.38536462, -0.22266881, -0.75220458, -0.20761696, -0.21995575,
        -1.66440488],
       [-0.47539902, -0.39926457,  0.4712879 ,  0.33475972, -0.91520702,
        -0.77392705,  0.0107958 , -0.59913616, -0.21606118,  0.23872552,
        -1.20191475, -0.33356105,  0.20716731, -0.87243656, -0.83738564,
        -1.51203164],
       [-0.77748537, -1.1011531 , -0.67013132, -0.17284687,  0.22838678,
        -0.47755088,  0.31938322, -0.73942682,  0.59003123,  0.32067039,
        -0.95389687,  0.42801456, -0.18414988,  0.43687449, -0.71045818,
        -0.49867805],
       [-0.57613817, -0.73859435, -0.07364678,  0.01579006,  0.06829177,
         0.1883118 ,  0.37773706, -0.25278286, -0.3451085 ,  0.18260311,
        -0.57528083, -0.32660089,  0.15631805, -0.48077116, -0.151794  ,
        -0.27274724],
       [ 0.40586482, -0.18762225,  0.06792857, -0.02756208, -0.88866507,
         0.32126385,  0.27231083,  0.60761859,  0.06701515,  0.1465858 ,
        -0.00734523, -0.87733991,  0.08475615, -0.50538885,  0.17476978,
        -0.37453744],
       [ 0.24776462,  0.30934198, -0.23430938,  0.03577179, -0.49276831,
         0.10327536,  0.54472029, -0.2109935 , -0.62203948, -0.39718593,
         0.04889539, -0.34976892, -0.12966643,  0.08707969,  0.01353127,
        -0.43985732],
       [ 0.11289386, -0.75320849,  0.28633133, -0.79734103, -0.09539013,
         0.37977743,  0.46837913, -0.09544513, -1.23588815,  0.13912751,
        -0.45828556, -0.26240935, -0.23999369, -0.79613507, -0.58436717,
        -0.24879591]]), 'b1': np.array([[-0.17896665,  0.11952543,  0.22844319,  0.49399462, -0.10006339,
         0.56908747,  0.08296823,  0.41556842,  0.55195769, -0.17944282,
        -0.79759928, -1.36035734, -0.21352279, -0.1600015 , -0.36659456,
         0.41122768]]), 'W2': np.array([[ 0.37294937, -0.1116447 ,  0.64810752,  0.21682071, -0.82626046,
         0.84141131, -0.14468229,  0.47451432, -0.26119673, -0.15366484,
        -0.2672082 ,  0.16503962, -0.75228174, -0.28190859, -0.43753453,
         0.11768875],
       [-0.8595535 ,  0.43579284,  0.22074319,  1.01085311, -0.06758291,
         0.38333678, -0.85044038, -0.23449069, -0.27034618,  0.30599985,
         0.5430643 , -0.36293024,  0.10019102,  0.94370936,  0.47093655,
        -0.56035674],
       [-0.05267911,  0.47095255, -1.05030252, -0.76257829, -0.99676709,
        -0.82617627, -0.13119642, -0.67723285,  0.15435077, -0.0634466 ,
        -0.40826556, -0.70659897, -0.68627669, -0.07131502, -0.21492472,
        -0.76339578],
       [-0.58808514, -0.39026512,  0.02979024, -0.44762049, -0.88955589,
         0.29954128, -0.88880725, -0.51089693,  0.95899888, -0.25449159,
        -0.96255097, -0.44440932,  0.43142776, -0.8927364 , -0.49948836,
        -0.77210218],
       [-0.79588357,  0.61831874,  0.58401464,  0.7719858 , -1.02045107,
         0.24629097,  0.75182192, -0.40009686,  0.00781769,  0.64182765,
         0.8745121 , -0.9962804 ,  0.57777517,  0.688603  , -0.88640975,
         0.04518056],
       [-1.15456204, -0.06881877, -0.76586844,  0.744277  , -1.05302313,
        -0.89237806, -0.69690849, -0.27082444,  0.05848458, -0.85595941,
         0.28976233,  0.033688  , -0.28207425, -1.06671545, -0.65748393,
        -0.18123273],
       [-0.03139966,  0.25838218, -0.84618477, -0.35205078, -0.63362573,
        -2.02721344, -1.00829618, -2.04826963,  0.23092128, -0.67299215,
        -0.33740387, -0.58131517, -2.25171464, -0.49611726, -0.30949862,
        -0.32112342],
       [-0.92958628,  0.02787528,  0.34596042, -0.18490562, -0.70379058,
         0.14233357, -0.0048928 ,  0.59035019,  0.05325069,  0.94947574,
        -0.87550419,  0.78562596,  0.55185998,  0.36068927, -0.24706695,
        -0.55715246],
       [-0.23898541, -0.37122901,  0.14737749,  0.54564523,  0.53877494,
         0.05363236, -0.39628557, -0.14605224,  0.18798835, -0.56874168,
        -0.06226102,  0.87153201, -0.13106095, -0.16727748,  0.0939399 ,
        -0.03363695],
       [ 0.17409919,  0.88294112,  0.54336304,  0.42554299,  0.88953659,
        -0.35401279,  0.1754984 , -0.71143696, -0.42721412,  0.79057019,
        -0.91388688, -0.51844209,  0.73812756, -0.01378564,  0.38809535,
        -0.61492343],
       [ 0.14324614, -0.6947471 , -0.3804144 , -0.88442544,  0.35309369,
        -0.46931101, -0.03112333, -0.31271859, -0.50564184,  0.30507603,
        -1.00775788,  0.39731855, -0.32460464, -0.3914421 , -0.70569881,
         0.35206825],
       [ 0.17788881, -0.12955643, -0.10846683, -0.83593863,  0.26202711,
        -0.47031339,  0.45680959, -0.71605531, -0.01552894, -0.76099882,
        -1.19902058, -0.03129861, -0.2640256 , -0.18161564, -0.15037789,
         0.37601758],
       [-0.39754858,  0.55958018,  0.76648752,  0.17649396, -0.31552746,
        -1.11966577,  0.26350039,  0.70611975, -0.24106752, -0.92954893,
         0.08526521, -0.16267175, -0.72194413, -0.74731192, -0.13410185,
        -0.6693976 ],
       [ 1.00995662, -0.08297511,  1.1152638 ,  0.97172346,  0.70122875,
        -0.80573675, -0.14628404, -0.10517784, -0.21530292,  0.45822885,
         0.26043507, -0.02898239,  0.15636155,  0.60556038, -0.6572015 ,
        -0.67717193],
       [ 0.85381191,  0.29182311, -0.33098604, -0.67769564,  0.53146811,
         0.74913508,  0.99091091,  0.91936825, -0.10736124, -0.67949067,
         0.56228581, -0.24872742,  0.47432919,  0.53009036,  0.42974098,
        -0.07688442],
       [ 0.79192654, -0.05627338, -0.41089001, -0.65764616, -0.13304841,
         0.47603343,  0.37621494,  0.29581963, -0.42231284,  0.14715744,
         0.40633267,  0.12755886, -0.09263157, -0.49094854,  0.73351138,
         0.66692818]]), 'b2': np.array([[ 0.62748731,  0.42693888, -0.1874175 ,  0.50062812,  0.40223401,
        -0.23778647,  0.5866824 ,  0.14090494, -0.16437188, -0.97443633,
        -1.24885273, -1.04223354,  0.19492054, -0.14879168,  0.24233177,
         0.04152686]]), 'W3': np.array([[ 0.26898617,  0.46267497, -0.13517043,  0.32767967],
       [ 0.27995893, -0.07921294, -0.31364408,  0.08307319],
       [ 0.82517234,  0.92057404, -0.13106051, -0.01739586],
       [-0.36671702, -0.20921794, -0.50557651, -0.45149484],
       [ 0.42820343,  0.28573798,  0.17923643,  0.07553329],
       [ 2.14908254, -1.06282469,  0.9367047 ,  0.1258717 ],
       [-0.24052079, -0.60643204,  0.0580624 ,  0.46340033],
       [-0.68952262,  0.84593351, -0.32538686, -0.55722258],
       [ 0.51621417,  0.18131603,  0.26618991, -0.35580265],
       [ 0.16517755, -0.13827393,  0.18073268,  0.4789714 ],
       [ 1.82358255, -1.82668508,  1.1887023 ,  1.1504594 ],
       [ 0.27537615, -0.56894293, -0.45583794,  0.24762381],
       [-0.67897382,  0.58123612, -0.76624386, -0.58136723],
       [-0.21498722,  1.63756641,  0.16856235,  0.90878896],
       [ 0.00275333,  0.005017  , -0.08533904,  0.6860622 ],
       [-0.14585374,  0.07934293,  0.18209333, -0.17981429]]), 'b3': np.array([[ 0.51900511, -0.09399733,  0.6588777 ,  0.49511561]])}


autre_bon_para = {'W1': np.array([[ 0.15325957, -0.3641428 , -1.26057415, -0.369454  , -0.3998766 ,
        -1.30052772, -0.35093732,  0.14248882,  0.96404674, -0.96646647,
        -1.15823987, -0.90719447, -0.31881358, -0.84142882, -0.88727552,
        -0.65488584],
       [ 0.62482717,  0.58046568, -0.84240198,  0.71486378,  0.55075188,
         0.15385222,  0.01729409,  0.69093515,  0.37394153,  0.02788618,
        -0.45707928,  0.27108947, -0.40991202, -0.03940321,  0.30736274,
         0.695053  ],
       [ 0.25671211, -0.87433034, -1.02359795, -0.5968132 , -0.35475047,
        -0.326949  , -0.02458198, -1.23512997, -0.97927745, -1.368306  ,
        -0.14305412,  0.66056656,  0.66972857, -0.4591556 , -0.7667926 ,
         0.47982302],
       [-0.28842879, -1.02903914,  0.15083052,  0.4696116 , -0.51390396,
        -1.18482861, -0.91683257, -0.06213557, -1.18569515,  0.14801165,
         0.01251544,  0.98796696, -0.6260862 ,  0.32122859, -0.51736663,
        -0.89276959],
       [ 0.06939462, -1.02938543,  0.35598645, -0.18910545, -0.36381159,
         0.76628658, -0.49089871,  0.1123979 ,  0.8766464 ,  0.25955805,
         0.29635203, -0.59385744, -0.72352494, -0.52717526, -0.49380601,
        -0.15055056],
       [ 0.17259427, -0.73491764,  0.10069482,  1.09419961, -0.30044849,
        -1.23798713, -0.58208637, -0.4641514 , -0.65824601,  0.24880926,
         0.08676307, -0.58673427, -0.99478449, -0.79436876,  0.51478073,
        -0.72309168],
       [ 0.22985844, -0.2405708 ,  0.57479885,  0.33233891, -0.95081612,
         0.49621073, -0.74046158, -0.94401676, -0.20011277, -0.43964127,
        -0.76261654,  0.58124538,  0.60993973, -0.99248073,  0.06624141,
         0.01542974],
       [ 0.99500498, -0.52745095,  0.49536864,  0.08121069, -0.53248943,
         0.06956708,  0.8738929 ,  0.20131755, -0.19207105, -0.05893841,
        -0.82538951, -1.02106292,  1.16667844, -1.00018809,  0.91430155,
         0.37629794],
       [-0.83644692, -0.11512131, -0.03212501,  0.45406574,  0.68375497,
        -0.11418564, -0.26874637, -0.16798082,  0.55412315, -0.0143486 ,
         0.59838191,  0.33373148, -0.38894011,  0.11151528,  0.02408935,
         0.1247989 ],
       [-1.35189623,  0.5219298 , -0.33472033, -0.23749202,  0.40223638,
         0.06464531,  0.52379999,  0.1159698 , -0.06614636,  0.4609972 ,
        -0.66754812,  0.05213019,  0.96575181,  0.12040338,  0.36932635,
         0.44801159],
       [-1.00518177, -0.6999734 , -1.18133418, -0.12238252, -0.68556471,
        -0.00147013, -0.37307515,  0.19037248, -0.2553659 , -0.30646799,
         0.4058866 , -0.4056129 ,  0.27630385,  0.39761949, -0.16513345,
         1.0989297 ],
       [-0.73903617,  0.22068481, -0.17969225,  0.23838311, -0.66178661,
         0.04314946,  0.46698499, -0.08603073,  0.61687812,  0.07333154,
         0.57827408,  0.02039519, -0.18352105, -0.34593183, -1.09260038,
         0.0473443 ]]), 'b1': np.array([[-0.47007439, -0.73102791,  0.83116497, -0.18112212, -0.30775082,
        -0.92435496,  1.05719662,  0.26816132,  0.04708368,  0.39174602,
         0.3203217 , -0.81991103, -0.94093639,  0.00364614, -0.70817379,
         0.37909058]]), 'W2': np.array([[-0.39566501,  1.02199444, -0.66269153,  0.02582887,  0.99037663,
         0.79574812,  0.42092138, -0.92535714],
       [ 0.40410208,  0.79403752,  0.43286158,  0.63405652,  0.39924565,
        -0.58085246,  0.8535929 ,  0.07784532],
       [-0.44062542, -0.10237124, -1.07883845, -0.30499172, -0.73291494,
        -1.00889188,  0.79912811,  0.10560769],
       [ 0.18653889, -0.40079749, -1.03466631, -0.72251934, -0.01535799,
         1.14446668,  0.55681976, -0.75122421],
       [-0.02339339, -0.56807334,  0.78363567,  0.59970989, -0.85652062,
         0.61739356,  0.67014211, -0.53512803],
       [-0.67202174, -0.47751452,  0.68684866, -0.50128966, -0.55810812,
        -0.40596655,  0.1443893 , -1.05009341],
       [ 0.14583554, -0.49087857,  0.50365381,  0.65676067, -0.54978325,
        -0.2901817 ,  0.16462012,  0.60155875],
       [ 1.21703188,  0.44613685,  0.25840272,  0.1550989 ,  0.78113309,
         0.61548331, -0.37059391, -0.61090434],
       [ 0.40389298,  0.0400418 , -0.85017953,  0.48657859, -0.55971803,
        -0.06698777,  0.41337523, -0.80298138],
       [-0.47164311,  0.01166668, -0.95477306, -0.74327655,  0.27011479,
        -0.51084682,  1.28337307,  0.06565169],
       [-0.25956049,  0.47396095,  0.20588516, -0.83178075, -0.85917357,
        -0.17435044,  0.08597016,  0.72195316],
       [ 0.24374738, -0.80497463,  0.02265144, -0.31270521,  0.38570093,
         0.0759441 ,  0.48715083, -0.93142501],
       [ 0.11643245,  0.01089114, -1.0354577 , -0.1202039 ,  0.02127222,
         0.34057085,  0.28167429, -0.01431237],
       [ 0.78202214,  0.2350574 , -0.68977222,  0.17830123,  0.77505275,
         0.54355179,  0.63966982, -0.38516283],
       [-0.93212662,  0.63032234,  0.70590791,  0.52248397, -0.47480872,
         0.12871366, -0.25667475, -0.56401158],
       [ 0.00402156, -0.88720535, -0.56344298, -0.22172775, -1.50259815,
        -0.20443345, -0.60803886, -0.1242442 ]]), 'b2': np.array([[ 0.88580327, -0.64787665,  0.54961113,  0.21120299, -0.79640382,
         0.64780553, -0.63953839,  0.01934063]]), 'W3': np.array([[-0.64825704, -0.74197005, -0.91341689, -0.51228272, -0.78354039,
        -1.29019207, -0.7437867 ,  0.4576919 ],
       [ 0.94893085, -1.00974308, -0.68438451,  0.16314807,  0.7753974 ,
         0.46029169, -0.82450593, -0.51761579],
       [ 0.57107139,  0.99561548,  0.8926242 , -0.44222597, -0.4739525 ,
         0.16432485, -0.01209482,  0.65940561],
       [ 0.41256465,  0.10962997, -0.56615546,  0.46421242,  0.49633273,
         0.30633935, -0.49008237, -0.42784057],
       [-0.28729589, -1.13422209,  0.82882133, -0.67074123,  0.78776773,
         0.55476704,  0.40662162, -0.42156015],
       [-0.01643575, -0.08216821, -0.29301929, -0.34840957, -0.71864975,
        -0.72208376, -0.31205608, -0.96744197],
       [-1.03299777,  0.29571278, -0.61492435, -0.94321258, -0.66637419,
        -0.08608567,  0.59697862,  0.06180396],
       [-0.50278958,  0.45652018, -0.3261176 , -0.3147911 ,  0.54526354,
        -0.84624132,  1.08479129,  0.71674732]]), 'b3': np.array([[-0.86466407,  0.57977031,  0.0698638 ,  0.13775766, -0.56076923,
         0.16551401,  0.24383723, -1.05385032]]), 'W4': np.array([[-0.38618976,  1.34326194, -0.37749661,  0.55052923],
       [ 0.07340672,  0.08118615, -0.36414211, -1.03994048],
       [-0.15300827,  0.87133661,  1.20609791, -0.32604493],
       [-0.09716398, -0.11441277,  0.47315629,  0.61402033],
       [ 0.36193507,  0.2271064 , -0.30101436,  0.53069639],
       [ 0.98746072,  0.3038177 ,  0.15928189, -0.65582187],
       [-0.10150531,  0.30884966, -0.23815826,  0.86553338],
       [-1.02446035, -0.81554583, -0.26034053, -0.52224178]]), 'b4': np.array([[0.42280072, 0.65680676, 0.59586334, 0.38843299]])}

bon_parametres = {'W1': np.array([[-0.52798256,  0.39269725,  0.50203016, -1.26809006,  0.35963562,
         0.56476246, -0.0382136 ,  1.23948535],
       [-0.08798542,  0.3378109 , -0.72615392,  0.18693983, -0.4235863 ,
         0.64271984, -0.23166281,  0.64805494],
       [ 0.58498918,  0.25358736, -0.85424781,  0.15338094,  1.1484595 ,
         0.0670273 ,  0.79701062,  1.05870151],
       [ 0.75434014, -0.13149729, -0.24132913, -1.40014737,  0.01230367,
         0.00207361, -0.90882952, -0.04902032],
       [ 0.58733198,  0.42185436, -0.47543131,  0.08912643,  1.01440599,
         0.06423055,  0.08542808,  0.15305099],
       [ 0.37962191, -1.12729526,  0.1950136 , -0.70424991,  0.08083505,
         0.7527454 , -0.45581309,  0.4051043 ],
       [ 0.28803255, -0.92348966, -0.72989344,  0.63253883,  0.0397583 ,
         0.61382195, -0.34698005,  0.14646403],
       [-0.14290791, -0.4056259 ,  0.72725076, -0.1782443 ,  1.24235425,
         0.31146393,  0.17897606,  0.59692721],
       [-0.26492794, -0.52796937,  0.22461175, -0.07106428, -0.03366546,
        -0.15742456,  0.57086679, -0.06609877],
       [-0.28589421,  0.29375911,  0.24295429,  0.17151378,  0.35518058,
         0.0085988 ,  0.03004663,  0.09710871],
       [-0.18700333, -0.11118268,  0.18818703, -0.01439862, -0.10158913,
        -0.16840262,  0.31412412,  0.10428259],
       [ 0.16727987,  0.00931452,  0.12953228, -0.93325905,  0.01164491,
         0.08079123, -0.26546531,  0.06547454]]), 'b1': np.array([[ 0.66364056,  0.53783215,  0.57470806,  0.08558139, -0.12010552,
        -0.75198154, -0.26682984, -0.5536961 ]]), 'W2': np.array([[-0.13580706,  0.06955813,  0.28185438, -0.05119879, -0.51987299,
        -0.72635673,  0.46360979,  0.35929626],
       [-0.74673596,  0.41003101, -0.15545228, -0.63714302, -0.46950394,
         0.83562567,  1.16369091,  0.05280155],
       [-0.03870511,  0.51343232,  0.69541224, -0.91607464,  0.30855034,
         0.01996909, -0.1023548 ,  0.88935077],
       [ 0.18490147, -0.73925385,  0.41939381, -0.20618024, -0.52997753,
        -0.59405845,  0.10447217,  1.21334722],
       [ 0.03509283, -1.07414104,  0.28576833, -1.51913349, -1.18644974,
        -0.26020483, -0.45556915, -0.04062462],
       [-0.99719735,  0.13426368,  0.56183754,  0.11271571,  0.33989311,
        -0.14156056, -1.0503853 ,  0.16638525],
       [-0.1535814 ,  0.14437798, -0.04513921, -0.8069609 ,  0.63282809,
         1.08948182, -0.17479002,  0.2713777 ],
       [-0.18028707, -0.12484333, -1.19560453,  0.05473821, -0.59630523,
        -0.50227852,  0.2054362 , -0.16286837]]), 'b2': np.array([[ 0.3625357 ,  1.23118533, -0.47833967,  0.44214474,  0.29382694,
        -0.1734796 , -0.46906729,  0.00644958]]), 'W3': np.array([[ 0.13038168, -0.22882789,  0.34164048, -0.52851685],
       [-0.34343994, -0.59072264,  0.10157924,  0.02070784],
       [ 0.04815991,  0.07204572, -0.93306965,  0.43532511],
       [ 0.66128534, -0.42146537,  0.09161201, -0.0526088 ],
       [-1.01014179,  1.06537599, -0.08226444,  0.03315401],
       [ 0.3135836 , -0.95968237, -0.93169733, -0.31983327],
       [-0.54235738,  0.2397026 ,  0.04289334,  0.09777898],
       [-0.00652767, -0.03324435,  0.13853847, -0.57287766]]), 'b3': np.array([[ 0.11514333,  0.53014935, -0.15600266,  0.50816511]])}

faire_jouer_IA(best_para)
faire_jouer_IA(bon_parametres)
faire_jouer_IA(autre_bon_para)  
