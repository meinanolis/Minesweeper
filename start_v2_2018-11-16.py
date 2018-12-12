import numpy as np 
from PIL import ImageTk, Image
import matplotlib.pyplot as plt 
import tkinter as tk

class game_operatoren:

	def __init__(self, defaultx, defaulty, defaultm):
		self.defaultx=defaultx
		self.defaulty=defaulty
		self.defaultm=defaultm

	def Spielumgebung(self):
		self.root = tk.Tk()
		self.root.title('Minesweeper')
		self.FrameMenu=tk.Frame(self.root)
		self.FrameMenu.grid(row=0,column=0)
		self.FrameMinefield=tk.Frame(self.root)
		self.FrameMinefield.grid(row=1,column=0)

		self.ButtonNewGame = tk.Button(self.FrameMenu, text='New Game',command=self.newgame)
		self.ButtonNewGame.grid(row=0,column=0)
		self.LabelNx =tk.Label(self.FrameMenu,text='Breite:')
		self.LabelNx.grid(row=1,column=0)
		self.EntryNx=tk.Entry(self.FrameMenu)
		self.EntryNx.grid(row=1,column=1)
		self.EntryNx.insert(tk.END, str(self.defaultx))
		self.LabelNy =tk.Label(self.FrameMenu,text='Höhe:')
		self.LabelNy.grid(row=2,column=0)
		self.EntryNy=tk.Entry(self.FrameMenu)
		self.EntryNy.grid(row=2,column=1)
		self.EntryNy.insert(tk.END, str(self.defaulty))
		self.LabelNm =tk.Label(self.FrameMenu,text='Minen:')
		self.LabelNm.grid(row=3,column=0)
		self.EntryNm=tk.Entry(self.FrameMenu)
		self.EntryNm.grid(row=3,column=1)
		self.EntryNm.insert(tk.END, str(self.defaultm))
		#AusblendenVar = tk.IntVar()
		#EntryAusblenden=tk.Checkbutton(FrameMenu, text='Ausblenden',variable=AusblendenVar, command=grid.spielfeld_anzeigen)
		#EntryAusblenden.grid(sticky="W", row=4,column=1)
		self.ShowSolutionVar = tk.IntVar()
		self.EntryShowSolution = tk.Checkbutton(self.FrameMenu, text='Show Solution',variable=self.ShowSolutionVar,command=self.HandleSettings)
		self.EntryShowSolution.grid(sticky="W", row=3,column=4)
		self.DetectInconclusivityVar=tk.IntVar()
		self.EntryDetectInconclusivity = tk.Checkbutton(self.FrameMenu, text='Detect Inconclusivity',variable=self.DetectInconclusivityVar, command=self.HandleSettings)
		self.EntryDetectInconclusivity.grid(sticky="W", row=0,column=4)
		self.AutoGuessVar=tk.IntVar()
		self.EntryAutoGuess = tk.Checkbutton(self.FrameMenu, text='Auto Guess',variable=self.AutoGuessVar)
		self.EntryAutoGuess.grid(sticky="W", row=2,column=4)
		self.AutoSolveVar=tk.IntVar()
		#EntryAutoSolve = tk.Checkbutton(FrameMenu, text='Auto Solve',variable=AutoSolveVar,command=AutoSolve)
		#EntryAutoSolve.grid(sticky="W", row=4,column=4)
		self.ButtonGuess = tk.Button(self.FrameMenu, text='Guess for me',command=self.Guess,state='disabled')
		self.ButtonGuess.grid(sticky="W", row=1,column=4)
		self.LabelRestminenText =tk.Label(self.FrameMenu,text='Restminen')
		self.LabelRestminenText.grid(row=0,column=5)
		self.LabelRestminen =tk.Label(self.FrameMenu,text='999')
		self.LabelRestminen.grid(row=1,column=5)
		self.LabelLösbarText =tk.Label(self.FrameMenu,text='Lösbar')
		self.LabelLösbarText.grid(row=2,column=5)
		self.LabelLösbar =tk.Label(self.FrameMenu,text='999')
		self.LabelLösbar.grid(row=3,column=5)

		self.button = tk.Button(self.root, text='Stop', width=25, command=self.root.destroy)
		self.button.grid(row=2,column=0)

	def newgame(self):
		nx=int(self.EntryNx.get())
		ny=int(self.EntryNy.get())
		m=float(self.EntryNm.get())
		self.grid=game_grid(nx,ny,m)
		self.grid.fill_in_mines()
		self.grid.bunte_nummern()
		self.grid.grundfeld_bild_erstellen()
		#self.grid.spielfeld_bild_erstellen()
		self.LabelField=tk.Label(self.FrameMinefield, text='bla')
		self.LabelField.bind("<Button-1>",lambda e:self.leftclick(e))
		self.LabelField.bind("<Button-3>",lambda e:self.rightclick(e))
		self.LabelField.bind("<Double-Button-1>",lambda e:self.doubleclick(e))
		self.LabelField.grid(row=0,column=0)
		self.spielfeld_anzeigen()

	def spielfeld_anzeigen(self):
		self.grid.spielfeld_bild_erstellen()
		self.im_spielfeld_tk=ImageTk.PhotoImage(self.grid.im_spielfeld)
		self.LabelField.configure(image=self.im_spielfeld_tk)
		self.LabelField.image=self.im_spielfeld_tk
		self.restminen=np.sum(self.grid.bombloc)-np.sum(self.grid.flagged_felder)
		self.LabelRestminen.configure(text=str(self.restminen))
		if np.array(self.grid.help_minen).any() or np.array(self.grid.help_sicher.any()):
			self.ButtonGuess.configure(state='disabled')
		else:
			self.ButtonGuess.configure(state='normal')
			if self.AutoGuessVar.get():
				print('autoguesss')
				self.Guess()

	def winner(self):
		print('winner')
		im=ImageTk.PhotoImage(self.grid.im_spielfeld_offen)
		self.LabelField.configure(image=im)
		self.LabelField.image=im
		self.LabelField.bind("<Button-1>",lambda e:self.newgame())

	def leftclick(self, e):
		x=int(e.x/self.grid.fieldsize)
		y=int(e.y/self.grid.fieldsize)
		print("leftclick!",x,y)
		if self.grid.flagged_felder[x,y]==1:
			self.grid.flagged_felder[x,y]=0
		else:
			self.aufdecken(x,y)
			self.spielfeld_anzeigen()
		if (self.grid.open_fields!=self.grid.bombloc).all():
			self.winner()
			
	def rightclick(self, e):
		x=int(e.x/self.grid.fieldsize)
		y=int(e.y/self.grid.fieldsize)
		print("rightclick!",x,y)
		self.grid.flagged_felder[x,y]=1
		self.grid.help_minen[x,y]=0
		self.grid.help_warsch[x,y]=np.nan
		self.spielfeld_anzeigen()
		if (self.grid.flagged_felder==self.grid.bombloc).all():
			self.winner()

	def doubleclick(self, e):
		x=int(e.x/self.grid.fieldsize)
		y=int(e.y/self.grid.fieldsize)
		print("doubleclick!",x,y)
		if self.grid.open_fields[x,y]==1:
			nflags=0
			for n in self.grid.nachbarfelder(x,y):
				nflags+=self.grid.flagged_felder[n[0],n[1]]
			if nflags==self.grid.field[x,y]:
				for n in self.grid.nachbarfelder(x,y):
					if self.grid.open_fields[n[0],n[1]]==0 and self.grid.flagged_felder[n[0],n[1]]==0:
						self.aufdecken(n[0],n[1])
						self.spielfeld_anzeigen()

	def aufdecken(self, x, y):
		self.grid.open_fields[x,y]=1
		#global save
		#save[x,y]=0
		
		if self.grid.bombloc[x,y]==0:
			if self.grid.field[x,y]==0:
				print('aufdecken1', x,y)
				for n in self.grid.nachbarfelder(x,y):
					if self.grid.open_fields[n[0],n[1]]==0:
						self.aufdecken(n[0],n[1])
			self.grid.help_sicher[x,y]=0
			self.grid.help_warsch[x,y]=np.nan
		else:
			print('looser')
			im=ImageTk.PhotoImage(self.grid.im_spielfeld_offen)
			self.LabelField.configure(image=im)
			self.LabelField.image=im
			self.LabelField.bind("<Button-1>",lambda e:self.newgame())

	def HandleSettings(self):
		if self.DetectInconclusivityVar.get():
			self.ButtonGuess.configure(state='normal')
			self.EntryShowSolution.configure(state='normal')
			self.EntryAutoGuess.configure(state='normal')
		else:
			self.ButtonGuess.configure(state='disabled')
			self.EntryShowSolution.configure(state='disabled')
			self.EntryAutoGuess.configure(state='disabled')
			self.AutoGuessVar.set(0)
		self.spielfeld_anzeigen()

	def Guess(self):
		if 0:#self.grid.help_warsch.any():
			pos=np.argwhere((self.grid.help_warsch+self.grid.bombloc)==np.nanmin(self.grid.help_warsch+self.grid.bombloc))[0]
			print(pos)
		elif (self.grid.u_rand-self.grid.flagged_felder==1).any():
			list_u_rand=np.argwhere(self.grid.u_rand-self.grid.bombloc==1)
			l=[]
			for ix,iy in list_u_rand:
				f=0
				for n in self.grid.nachbarfelder(ix,iy):
					f+=self.grid.open_fields[n[0],n[1]]
					f+=self.grid.flagged_felder[n[0],n[1]]
				l.append(f)
			pos=list_u_rand[np.argmax(l)]
		else:
			list_=np.argwhere(1-self.grid.open_fields-self.grid.bombloc==1)
			rando=np.random.random_integers(0,len(list_)-1,1)[0]
			pos=list_[rando]
		self.aufdecken(pos[0],pos[1])
		self.spielfeld_anzeigen()
		if (self.grid.open_fields!=self.grid.bombloc).all():
			self.winner()


class game_grid: 

	fieldsize=32
	
	def __init__(self, size_x, size_y, n_mines_fraction):
		self.size_x = size_x
		self.size_y = size_y
		self.n_mines_fraction = n_mines_fraction
		self.n_mines = np.round(size_x*size_y*n_mines_fraction)
		self.open_fields=np.reshape(np.zeros(self.size_x*self.size_y),(self.size_x,self.size_y))
		self.flagged_felder=np.zeros_like(self.open_fields)
		self.help_minen=np.zeros_like(self.open_fields)
		self.help_warsch=np.zeros_like(self.open_fields)
		self.help_warsch[:]=np.nan
		self.help_sicher=np.zeros_like(self.open_fields)
		self.b_rand=np.zeros_like(self.open_fields)
		self.u_rand=np.zeros_like(self.open_fields)

	def fill_in_mines(self):
		n=0
		bombloc=np.zeros(self.size_x*self.size_y)
		while n<self.n_mines:
			rand_pos=np.random.random_integers(0,self.size_x*self.size_y-1,1)
			bombloc[rand_pos]=1
			n=np.sum(bombloc)
		self.bombloc=np.reshape(bombloc,(self.size_x,self.size_y))

	def nachbarfelder(self, x, y):
		alle_nachbarn=[[0,1],[0,-1],[1,0],[-1,0],[1,-1],[-1,-1],[-1,1],[1,1]]
		alle_nachbarn_ohne_rand=[]
		for n in alle_nachbarn:
			if x+n[0]>=0 and y+n[1]>=0 and x+n[0]<self.size_x and y+n[1]<self.size_y:
				alle_nachbarn_ohne_rand.append([x+n[0],y+n[1]])
		return alle_nachbarn_ohne_rand

	def bunte_nummern(self):
		self.field=np.zeros_like(self.open_fields)
		for x in range(self.size_x):
			for y in range(self.size_y):
				if self.bombloc[x,y]:
					self.field[x,y]=np.nan
				else:
					for n in self.nachbarfelder(x,y):
						self.field[x,y]+=self.bombloc[n[0],n[1]]

	def grundfeld_bild_erstellen(self):
		im=Image.new('RGB',(self.size_x*self.fieldsize,self.size_y*self.fieldsize))
		####place numbers
		for i in range(0,9,1):
			if i == 0:
				im_number=Image.open('Bilder/blank.png').resize((self.fieldsize,self.fieldsize))
			else:
				im_number=Image.open('Bilder/'+str(i)+'.png').resize((self.fieldsize,self.fieldsize))
			for ix,iy in np.argwhere(self.field==i):
				im.paste(im_number,(ix*self.fieldsize,iy*self.fieldsize))
		####place Bombs
		im_bomb=Image.open('Bilder/bomb.png').resize((self.fieldsize,self.fieldsize))
		for ix,iy in np.argwhere(self.bombloc==1):
			im.paste(im_bomb,(ix*self.fieldsize,iy*self.fieldsize))
		im.save('temp_game.png')
		self.im_spielfeld_offen=im

	def spielfeld_bild_erstellen(self):
		im_unknown	=	Image.open('Bilder/unknown.png').resize((self.fieldsize,self.fieldsize))
		im_flag		=	Image.open('Bilder/flag.png').resize((self.fieldsize,self.fieldsize))
		#unbekannte felder verstecken
		self.im_spielfeld=self.im_spielfeld_offen.copy()
		for ix,iy in np.argwhere(self.open_fields==0):
			self.im_spielfeld.paste(im_unknown,(ix*self.fieldsize,iy*self.fieldsize))
		#Fahnen setzen
		for ix,iy in np.argwhere(self.flagged_felder==1):
			self.im_spielfeld.paste(im_flag,(ix*self.fieldsize,iy*self.fieldsize))
		
		if game.DetectInconclusivityVar.get():
			self.raender()
			self.simple_help()
			if game.ShowSolutionVar.get():
				im_save=Image.open('Bilder/green.png').resize((self.fieldsize,self.fieldsize))
				for ix,iy in np.argwhere(self.help_sicher):
					self.im_spielfeld.paste(im_save,(ix*self.fieldsize,iy*self.fieldsize))
				
				im_unsave=Image.open('Bilder/red.png').resize((self.fieldsize,self.fieldsize))
				for ix,iy in np.argwhere(self.help_minen):
					self.im_spielfeld.paste(im_unsave,(ix*self.fieldsize,iy*self.fieldsize))

	def raender(self):
		#b_rand
		self.b_rand=np.zeros_like(self.open_fields)
		for ix,iy in np.argwhere(self.open_fields):
			for n in self.nachbarfelder(ix,iy):
				if not self.open_fields[n[0],n[1]]:
					self.b_rand[ix,iy]=1
					break
		#u_rand
		self.u_rand=np.zeros_like(self.open_fields)
		for ix,iy in np.argwhere(self.b_rand):
			for n in self.nachbarfelder(ix,iy):
				if not self.open_fields[n[0],n[1]]:
					self.u_rand[n[0],n[1]]=1

	def simple_help(self):
		game.LabelLösbar.configure(text=str(int(np.sum(self.help_minen)+np.sum(self.help_sicher))))
		if not (np.array(self.help_minen).any() or np.array(self.help_sicher.any())):
			print('simple_help')
			for ix,iy in np.argwhere(self.b_rand): #für jedes bunte feld am rand
				wert=self.field[ix,iy]
				flag=0
				u_field=[]
				for n in self.nachbarfelder(ix,iy):
					if self.open_fields[n[0],n[1]]:
						pass 
					elif self.flagged_felder[n[0],n[1]]:
						flag+=1
					else:
						u_field.append(n)
				if wert-flag==0:
					for ix,iy in u_field:
						self.help_sicher[ix,iy]=1
						self.help_warsch[ix,iy]=0
				if wert-flag==len(u_field):
					for ix,iy in u_field:
						self.help_minen[ix,iy]=1
						self.help_warsch[ix,iy]=1
			game.LabelLösbar.configure(text=str(int(np.sum(self.help_minen)+np.sum(self.help_sicher))))
			if not (np.array(self.help_minen).any() or np.array(self.help_sicher.any())):
				print('extended_help')
				self.extended_help()

	def extended_help(self):
		list_u_rand=np.argwhere(self.u_rand-self.flagged_felder==1)
		pack_size=10 #größe der päckchen
		help_pack_list=[]
		while len(list_u_rand)>=pack_size:
			help_pack_list.append(help_pack(list_u_rand[:pack_size]))
			list_u_rand=list_u_rand[pack_size:]
		help_pack_list.append(help_pack(list_u_rand))

		for p in help_pack_list:
			all_bin_array=p.create_bin_vectors()
			for bin_array in all_bin_array:
				p.check_consistency(bin_array)
		
		#Lösungen aus packs zusammensetzten
		list_u_rand=np.argwhere(self.u_rand-self.flagged_felder==1)
		all_pack=help_pack(list_u_rand)
		N=1
		N2=1
		for pack in help_pack_list:
			pack.N=len(pack.consistant_bin_arrays)
			N=N*pack.N
			N2=N2*(2**pack.len)
		print('helper needs '+str(N)+'('+str(int(N/N2*100))+'%) instead of '+ str(N2)+' runs')
		i_vec=np.zeros([len(help_pack_list)])
		for i in range(N):
			if N>200000:
				print('N zu groß :(')
				break
			game.root.update()
			combined_bin_array=np.zeros_like(self.open_fields)
			for j,pack in zip(range(len(i_vec)),help_pack_list):
				combined_bin_array=combined_bin_array+pack.consistant_bin_arrays[int(i_vec[j])]
			for j in range(len(i_vec)):
				if i_vec[j]+1<help_pack_list[j].N:
					i_vec[j]+=1
					break
				else:
					i_vec[j]=0
			if np.sum(combined_bin_array)<=game.restminen:
				all_pack.check_consistency(combined_bin_array,True)
		print('and found '+str(len(all_pack.consistant_bin_arrays))+' possible solutions')
		game.grid.help_warsch=np.zeros_like(game.grid.open_fields)
		game.grid.help_minen=np.ones_like(game.grid.open_fields)
		for a in all_pack.consistant_bin_arrays:
			game.grid.help_minen=np.logical_and(game.grid.help_minen,a)
			game.grid.help_warsch+=a
		game.grid.help_sicher=game.grid.u_rand.copy()
		game.grid.help_sicher=np.logical_xor(game.grid.help_sicher,game.grid.flagged_felder)
		game.grid.help_warsch[game.grid.u_rand==0]=np.nan
		game.grid.help_warsch=game.grid.help_warsch/len(all_pack.consistant_bin_arrays)
		
		for a in all_pack.consistant_bin_arrays:
			game.grid.help_sicher=np.logical_and(game.grid.help_sicher,1-a)
		game.LabelLösbar.configure(text=str(int(np.sum(self.help_minen)+np.sum(self.help_sicher))))
		
		if 0:	###########plot
			plot=game.grid.help_warsch
			plt.imshow(plot.T)
			plt.show()



class help_pack:

	def __init__(self, pos_list):
		self.consistant_bin_vectors=[]
		self.consistant_bin_arrays=[]
		self.pos_list=pos_list
		self.len=len(pos_list)
		self.pos_array=np.zeros_like(game.grid.open_fields)
		self.open_nachbarfeld_array=np.zeros_like(game.grid.open_fields)
		for x,y in pos_list:
			self.pos_array[x,y]=1
			for nx,ny in game.grid.nachbarfelder(x,y):
				if game.grid.open_fields[nx,ny]:
					self.open_nachbarfeld_array[nx,ny]=1
		self.open_nachbarfelder=[]
		for x,y in np.argwhere(self.open_nachbarfeld_array):
			self.open_nachbarfelder.append([x,y])

	def check_consistency(self, bin_array, hard=False):
		consist=True
		for x,y in self.open_nachbarfelder:
			fieldvalue=game.grid.field[x,y]
			for nx,ny in game.grid.nachbarfelder(x,y):
				fieldvalue-=bin_array[nx,ny]
				fieldvalue-=game.grid.flagged_felder[nx,ny]
			if fieldvalue and hard:
				consist=False
				break
			if fieldvalue<0:
				consist=False
				break

		if consist:
			self.consistant_bin_arrays.append(bin_array)

	def create_bin_vectors(self):
		all_bin_array=[]
		for i in range(2**self.len):
			binär=np.binary_repr(i)
			while len(binär)<self.len:
				binär='0'+binär
			bin_vector=[int(b) for b in binär]
			bin_array=np.zeros_like(game.grid.open_fields)
			for p,b in zip(self.pos_list,bin_vector):
				bin_array[p[0],p[1]]=b
			all_bin_array.append(bin_array)
		return all_bin_array
			








game=game_operatoren(20,10,.3)
game.Spielumgebung()
game.newgame()

game.root.mainloop()