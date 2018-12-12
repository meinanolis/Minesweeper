import tkinter as tk
import numpy as np 
from PIL import ImageTk, Image
import matplotlib.pyplot as plt 
#from functions import *
import pandas as pd
import time

defaultx=20
defaulty=7
defaultm=.3
fieldsize=32
record_game=False
automode=0
max_fields_opend=100
save=np.zeros(defaultx*defaulty)
save=np.reshape(save,(defaultx,defaulty))
unsave=np.zeros_like(save)
mögliche_bombenanordnungen=np.array([[],[]])


def FillInMines(x,y,m):
	nbombs=int(x*y*m)
	n=0
	bombloc=np.zeros(x*y)
	while n<nbombs:
		for i in np.random.random_integers(0,x*y-1,1):
			bombloc[i]=1
		n=np.sum(bombloc)
		#print(n)
	bombloc=np.reshape(bombloc,(x,y))
	return bombloc

def BunteNummern(x,y,bombloc):
	#print('bombloc',bombloc)
	field=np.zeros(x*y)
	field=np.reshape(field,(x,y))
	for ix in range(x):
		for iy in range(y):
			for n in [[0,1],[0,-1],[1,0],[-1,0],[1,-1],[-1,-1],[-1,1],[1,1]]:
				if ix+n[0]>=0 and iy+n[1]>=0 and ix+n[0]<x and iy+n[1]<y:
					field[ix,iy]+=bombloc[ix+n[0],iy+n[1]]
	#print('BunteNummern',field)
	return(field)

def AssambleImage(x,y,field,bombloc):
	im=Image.new('RGB',(x*fieldsize,y*fieldsize))
	####place numbers
	for i in range(0,9,1):
		if i == 0:
			im_number=Image.open('Bilder/blank.png').resize((fieldsize,fieldsize))
		else:
			im_number=Image.open('Bilder/'+str(i)+'.png').resize((fieldsize,fieldsize))
		for ix,iy in np.argwhere(field==i):
			im.paste(im_number,(ix*fieldsize,iy*fieldsize))
	####place Bombs
	im_bomb=Image.open('Bilder/bomb.png').resize((fieldsize,fieldsize))
	for ix,iy in np.argwhere(bombloc==1):
		im.paste(im_bomb,(ix*fieldsize,iy*fieldsize))
	im.save('temp_game.png')
	return im

def newgame():
	x=int(EntryNx.get())
	y=int(EntryNy.get())
	m=float(EntryNm.get())
	global save
	global unsave
	save=np.zeros(x*y)
	save=np.reshape(save,(x,y))
	unsave=np.zeros_like(save)
	global bombloc
	bombloc=FillInMines(x,y,m)
	#np.save('bombloc',bombloc)
	global field
	field=BunteNummern(x,y,bombloc)
	#np.save('field',field)
	im=AssambleImage(x,y,field,bombloc)
	#open_fields
	global open_fields
	open_fields=np.zeros(x*y)
	open_fields=np.reshape(open_fields,(x,y))
	#np.save('open_fields',open_fields)
	#relevant_fields
	global relevant_fields
	relevant_fields=np.ones(x*y)
	relevant_fields=np.reshape(relevant_fields,(x,y))
	#np.save('relevant_fields',relevant_fields)
	#flagged_fields
	global flagged_fields
	flagged_fields=np.zeros(x*y)
	flagged_fields=np.reshape(flagged_fields,(x,y))
	#np.save('flagged_fields',flagged_fields)
	global LabelField
	LabelField=tk.Label(FrameMinefield, text='bla')
	LabelField.bind("<Button-1>",lambda e:leftclick(e))
	LabelField.bind("<Button-3>",lambda e:rightclick(e))
	LabelField.bind("<Double-Button-1>",lambda e:doubleclick(e))
	LabelField.grid(row=0,column=0)
	
	spielfeld_anzeigen()
	HandleSettings()
	return im
	#plt.imshow(im)
	#plt.xticks([32+i*fieldsize for i in range(x)],[i+1 for i in range(x)])
	#plt.yticks([32+i*fieldsize for i in range(y)],[i+1 for i in range(y)])
	#plt.show()

def Ausblenden(open_fields,flagged_fields):
	global relevant_fields
	open_fields_pos=np.argwhere(np.logical_and(open_fields,relevant_fields))
	mx=open_fields.shape[0]
	my=open_fields.shape[1]
	#print('open field pos',open_fields_pos)
	Ausblenden=[]
	for pos in open_fields_pos:
		all_neighbors=pos+np.array([[0,1],[0,-1],[1,0],[-1,0],[1,-1],[-1,-1],[-1,1],[1,1]])
		#Räder ausschließen
		#print('position',pos)
		yRand=np.isin(all_neighbors[:,1],[my,-1],invert=True)
		xRand=np.isin(all_neighbors[:,0],[mx,-1],invert=True)
		Rand=np.logical_and(xRand,yRand)
		#print('alle Nachbarn ohne Rand',all_neighbors[Rand]) #alle Nachbarn ohne Rand
		ausblend=np.array([(open_fields[i[0],i[1]]==1 or flagged_fields[i[0],i[1]]==1) for i in all_neighbors[Rand]])
		#ausblend=np.array([np.isin(open_fields_pos, i).all(1).any() for i in all_neighbors[Rand]])
		#print('sind nachbarn offen',ausblend)
		if ausblend.all():
			relevant_fields[pos[0],pos[1]]=0
	#print('relevant_fields',relevant_fields)
	#plt.imshow(relevant_fields.T-flagged_fields.T)
	#plt.show()
		
def spielfeld_anzeigen():
	im=Image.open('temp_game.png')
	#ungeöffnete Felder verdecken
	global open_field#s=np.load('open_fields.npy')
	global relevant_fields#=np.load('relevant_fields.npy')
	im_unknown=Image.open('Bilder/unknown.png').resize((fieldsize,fieldsize))
	for ix,iy in np.argwhere(open_fields==0):
		im.paste(im_unknown,(ix*fieldsize,iy*fieldsize))
	#Fahnen setzen
	global flagged_fields
	im_flagged=Image.open('Bilder/flag.png').resize((fieldsize,fieldsize))
	for ix,iy in np.argwhere(flagged_fields==1):
		im.paste(im_flagged,(ix*fieldsize,iy*fieldsize))
	if DetectInconclusivityVar.get():
		global save
		global unsave
		save,unsave = helpme()
		if ShowSolutionVar.get():
			im_save=Image.open('Bilder/green.png').resize((fieldsize,fieldsize))
			for ix,iy in np.argwhere(save==1):
				im.paste(im_save,(ix*fieldsize,iy*fieldsize))
			
			im_unsave=Image.open('Bilder/red.png').resize((fieldsize,fieldsize))
			for ix,iy in np.argwhere(unsave==1):
				im.paste(im_unsave,(ix*fieldsize,iy*fieldsize))
		global ButtonGuess
		if save.any() or unsave.any():
			ButtonGuess.configure(state='disabled')
		else:
			ButtonGuess.configure(state='normal')
			if AutoGuessVar.get():
				print('autoguesss')
				Guess()


	if AusblendenVar.get():
		global relevant_fields
		Ausblenden(open_fields,flagged_fields)
		im_blank=Image.open('Bilder/blank.png').resize((fieldsize,fieldsize))
		for ix,iy in np.argwhere(relevant_fields==0):
			im.paste(im_blank,(ix*fieldsize,iy*fieldsize))
	global FrameMinefield
	global LabelField
	global im2
	#im_field=tk.PhotoImage(file="temp_game.png")
	#LabelField.configure(image=im_field)

	im2=ImageTk.PhotoImage(im)
	LabelField.configure(image=im2)
	
	global bombloc#=np.load('bombloc.npy')
	restminen=np.sum(bombloc)-np.sum(flagged_fields)
	LabelRestminen.configure(text=str(restminen))
	
def leftclick(e):
	x=int(e.x/fieldsize)
	y=int(e.y/fieldsize)
	print("leftclick!",x,y)
	global flagged_fields#=np.load('flagged_fields.npy')
	if flagged_fields[x,y]==1:
		flagged_fields[x,y]=0
		#np.save('flagged_fields',flagged_fields)
	else:
		global open_fields#=np.load('open_fields.npy')
		global bombloc#=np.load('bombloc.npy')
		global field#=np.load('field.npy')
		aufdecken(x,y,bombloc,field)
		#if np.argwhere(open_fields).shape[0]>300:
		#	get_train_data(x,y,open_fields,bombloc,field)
	spielfeld_anzeigen()

	if (open_fields!=bombloc).all():
		print('winner')
		im=Image.open('temp_game.png')
		global im2
		im2=ImageTk.PhotoImage(im)
		LabelField.configure(image=im2)
		LabelField.bind("<Button-1>",lambda e:newgame())

def leftclick_Auto(open_fields,bombloc,field):
	if (open_fields+bombloc==1).all():
		newgame()
	else:
		global save
		global unsave
		#global flagged_fields
		if (unsave).any():
			p=np.argwhere(unsave)[-1]
			print("rightclick!",p[0],p[1])
			flag(p[0],p[1])
			#aufdecken(x,y,bombloc,field)
			spielfeld_anzeigen()
		elif(save).any():
			p=np.argwhere(save)[-1]
			print("leftclick!",p[0],p[1])
			aufdecken(p[0],p[1],bombloc,field)
			spielfeld_anzeigen()
		else:
			global automode
			automode=0
			AutoSolveVar.set(0)
			#print(open_fields+bombloc)
	global max_fields_opend
	if np.argwhere(open_fields).shape[0]>max_fields_opend and record_game:
		get_train_data(x,y,open_fields,bombloc,field)
		global defaultm
		defaultm=np.random.random()/3
		max_fields_opend=np.random.randint(500)
		global im
		im=newgame()

def Guess():
	global listuRand
	global mögliche_bombenanordnungen,bombloc,field,open_fields
	print('guess')
	#print(mögliche_bombenanordnungen)
	#guess_save=[]
	#for i in range(len(mögliche_bombenanordnungen[0,:])):
	#	if open_fields.any() :
	#		guess_save.append(np.sum(mögliche_bombenanordnungen[:,i]))
	#if len(guess_save):
	#	pos=listuRand[np.argmin(guess_save)]
	#else:
		#open the field with the most open neibors
	if not open_fields.any():
		pos=[0,0]
		pos[0]=np.random.random_integers(0,len(field[:,0])-1,1)[0]
		pos[1]=np.random.random_integers(0,len(field[0,:])-1,1)[0]
	elif not len(listuRand):
		pos=[0,0]
		pos[0]=np.random.random_integers(0,len(field[:,0])-1,1)[0]
		pos[1]=np.random.random_integers(0,len(field[0,:])-1,1)[0]
	else:
		l=[]
		mx=open_fields.shape[0]
		my=open_fields.shape[1]
		for pos in listuRand:
			all_neighbors=pos+np.array([[0,1],[0,-1],[1,0],[-1,0],[1,-1],[-1,-1],[-1,1],[1,1]])
			yRand=np.isin(all_neighbors[:,1],[my,-1],invert=True)
			xRand=np.isin(all_neighbors[:,0],[mx,-1],invert=True)
			Rand=np.logical_and(xRand,yRand)
			all_neighbors=all_neighbors[Rand]
			f=0
			for n in all_neighbors:
				f+=open_fields[n[0],n[1]]
			l.append(f)
		pos=listuRand[np.argmax(l)]
	if bombloc[pos[0],pos[1]]:
		flag(pos[0],pos[1])
	else:
		aufdecken(pos[0],pos[1],bombloc,field)
	spielfeld_anzeigen()
	print('aufdecken', pos)	

def doubleclick(e):
	x=int(e.x/fieldsize)
	y=int(e.y/fieldsize)
	print("doubleclick!",x,y)
	global open_fields#=np.load('open_fields.npy')
	global bombloc#=np.load('bombloc.npy')
	global flagged_fields#=np.load('flagged_fields.npy')
	global field#=np.load('field.npy')
	mx=flagged_fields.shape[0]
	my=flagged_fields.shape[1]
	if open_fields[x,y]==1:
		#open_fields=np.load('open_fields.npy')
		#bombloc=np.load('bombloc.npy')
		#field=np.load('field.npy')
		nflags=0
		for n in [[0,1],[0,-1],[1,0],[-1,0],[1,-1],[-1,-1],[-1,1],[1,1]]:
			if x+n[0]>=0 and y+n[1]>=0 and x+n[0]<mx and y+n[1]<my:
				nflags+=flagged_fields[x+n[0],y+n[1]]
		if nflags==field[x,y]:
			for n in [[0,1],[0,-1],[1,0],[-1,0],[1,-1],[-1,-1],[-1,1],[1,1]]:
				if x+n[0]>=0 and y+n[1]>=0 and x+n[0]<mx and y+n[1]<my:
					nflags+=flagged_fields[x+n[0],y+n[1]]
					if open_fields[x+n[0],y+n[1]]==0 and flagged_fields[x+n[0],y+n[1]]==0:
						aufdecken(x+n[0],y+n[1],bombloc,field)
	else:
		aufdecken(x,y,bombloc,field)
	spielfeld_anzeigen()

def rightclick(e):
	x=int(e.x/fieldsize)
	y=int(e.y/fieldsize)
	print("rightclick!",x,y)
	flag(x,y)
	spielfeld_anzeigen()
	global flagged_fields#np.load('flagged_fields.npy')
	global bombloc#=np.load('bombloc.npy')
	if (flagged_fields==bombloc).all():
		print('winner')
		im=Image.open('temp_game.png')
		global im2
		im2=ImageTk.PhotoImage(im)
		LabelField.configure(image=im2)
		LabelField.bind("<Button-1>",lambda e:newgame())

def aufdecken(x,y,bombloc,field):
	global open_fields
	open_fields[x,y]=1
	global save
	save[x,y]=0
	
	if bombloc[x,y]==0:
		
		mx=field.shape[0]
		my=field.shape[1]
		if field[x,y]==0:
			#print('aufdecken1')
			pos=np.array([x,y])
			print('aufdecken1', pos)
			all_neighbors=pos+np.array([[0,1],[0,-1],[1,0],[-1,0],[1,-1],[-1,-1],[-1,1],[1,1]])
			#Räder ausschließen
			#print('position',pos)
			yRand=np.isin(all_neighbors[:,1],[my,-1],invert=True)
			xRand=np.isin(all_neighbors[:,0],[mx,-1],invert=True)
			Rand=np.logical_and(xRand,yRand)
			#print('alle Nachbarn ohne Rand',all_neighbors[Rand]) #alle Nachbarn ohne Rand
			for n in all_neighbors[Rand]:
				if open_fields[n[0],n[1]]==0:
					aufdecken(n[0],n[1],bombloc,field)
					#print('aufdecken3',x+n[0],y+n[1])
		
	else:
		print('looser')
		im=Image.open('temp_game.png')
		global im2
		global LabelField
		im2=ImageTk.PhotoImage(im)
		LabelField.configure(image=im2)
		LabelField.bind("<Button-1>",lambda e:newgame())

def get_train_data(x,y,open_fields,bombloc,field):
	#get fields to train with. der rand von open Fields wird ermittelt
	mx=field.shape[0]
	my=field.shape[1]
	erweitertes_open_fields=np.zeros_like(open_fields)
	for pos in np.argwhere(open_fields):
		all_neighbors=pos+np.array([[0,1],[0,-1],[1,0],[-1,0],[1,-1],[-1,-1],[-1,1],[1,1]])
		yRand=np.isin(all_neighbors[:,1],[my,-1],invert=True)
		xRand=np.isin(all_neighbors[:,0],[mx,-1],invert=True)
		Rand=np.logical_and(xRand,yRand)
		all_neighbors=all_neighbors[Rand]
		for p in all_neighbors:
			erweitertes_open_fields[p[0],p[1]]=1

	alle_bekannten_kacheln=open_fields*field#+(1-open_fields)*9
		#plt.imshow(alle_kacheln)
		#plt.show()
		#break
	lern_kacheln=erweitertes_open_fields-open_fields
	#plt.imshow(lern_kacheln)
	#plt.show()
	alle_bekannten_kacheln=alle_bekannten_kacheln.reshape(720)
	global df_learn
	global columns
	for pos in np.argwhere(lern_kacheln):
		poi=np.zeros_like(lern_kacheln)
		poi[pos[0],pos[1]]=1
		poi=poi.reshape(720)
		open_fields_flat=open_fields.reshape(720)
		featurelist=np.vstack((poi,open_fields_flat,alle_bekannten_kacheln)).T.reshape(2160)
		featurelist=np.hstack((featurelist,bombloc[pos[0],pos[1]]))
		df_learn_temp=pd.DataFrame(featurelist.reshape((1,2161)),columns=columns)
		
		df_learn=pd.concat((df_learn,df_learn_temp),ignore_index=True)
	with open('train_data.csv', 'a') as f:
		df_learn.to_csv(f, header=False)
	df_learn=pd.DataFrame([],columns=columns)
	print('df_learn gespeichert', len(df_learn.index))

def flag(x,y):
	global flagged_fields
	global unsave
	flagged_fields[x,y]=1
	unsave[x,y]=0
	#np.save('flagged_fields',flagged_fields)

def helpme():
	if 0: #experimentell
		helpme2()
		#return
	print('help')
	global field
	global open_fields
	global flagged_fields
	global save
	global unsave
	if (unsave).any() or (save).any():
		return save, unsave
	mx=field.shape[0]
	my=field.shape[1]
	verringertes_open_fields=np.zeros_like(open_fields)
	for pos in np.argwhere(open_fields==0):
		all_neighbors=pos+np.array([[0,1],[0,-1],[1,0],[-1,0],[1,-1],[-1,-1],[-1,1],[1,1]])
		yRand=np.isin(all_neighbors[:,1],[my,-1],invert=True)
		xRand=np.isin(all_neighbors[:,0],[mx,-1],invert=True)
		Rand=np.logical_and(xRand,yRand)
		all_neighbors=all_neighbors[Rand]
		for p in all_neighbors:
			verringertes_open_fields[p[0],p[1]]=1
	bRand=open_fields-verringertes_open_fields #bunter Rand ist wo array 0 ist

	verringertes_open_fields=np.zeros_like(open_fields)
	for pos in np.argwhere(open_fields==1):
		all_neighbors=pos+np.array([[0,1],[0,-1],[1,0],[-1,0],[1,-1],[-1,-1],[-1,1],[1,1]])
		yRand=np.isin(all_neighbors[:,1],[my,-1],invert=True)
		xRand=np.isin(all_neighbors[:,0],[mx,-1],invert=True)
		Rand=np.logical_and(xRand,yRand)
		all_neighbors=all_neighbors[Rand]
		for p in all_neighbors:
			verringertes_open_fields[p[0],p[1]]=1
	uRand=open_fields-verringertes_open_fields #verdeckter Rand ist wo array 0 ist

	for pos in np.argwhere(bRand==0): #für jedes bunte feld am rand
		all_neighbors=pos+np.array([[0,1],[0,-1],[1,0],[-1,0],[1,-1],[-1,-1],[-1,1],[1,1]])
		yRand=np.isin(all_neighbors[:,1],[my,-1],invert=True)
		xRand=np.isin(all_neighbors[:,0],[mx,-1],invert=True)
		Rand=np.logical_and(xRand,yRand)
		all_neighbors=all_neighbors[Rand] #alle nachbarfelder ohne Spielfeldrand
		square=np.zeros((3,3))
		for p in all_neighbors: #p ist die echte position im spielfeld, sp ist die position im 3x3Feld
			sp=p-pos+np.array([1,1])
			if open_fields[p[0],p[1]]:
				square[sp[0],sp[1]]=0
			elif flagged_fields[p[0],p[1]]:
				square[sp[0],sp[1]]=-1
			else:
				square[sp[0],sp[1]]=1
		#print(square)
		#plt.imshow(square.T)
		#plt.show()
		kachelwert=field[pos[0],pos[1]]
		nnflags=len(np.argwhere(square.reshape(9)==-1))
		nclosed=len(np.argwhere(square.reshape(9)==1))
		#print(kachelwert,nnflags,nclosed)
		if kachelwert-nnflags==nclosed:
			unsavepos=np.argwhere(square==1)+pos-np.array([1,1])
			for pp in unsavepos:
				unsave[pp[0],pp[1]]=1
		if kachelwert==nnflags:
			savepos=np.argwhere(square==1)+pos-np.array([1,1])
			for pp in savepos:
				save[pp[0],pp[1]]=1
	if save.any() or unsave.any():
		return save, unsave
	else:
		global listuRand
		listuRand=np.argwhere(uRand+flagged_fields==-1)
		n=len(listuRand) # anzahl der unbekannten Randfelder
		I=2**n
		print('anzahl der unbekannten Randfelder:',n,'Anzahl der Bombenkombinationen',I)
		#print(uRand)
		#plt.imshow(uRand.T)
		#plt.show()
		if n>13:
			print('zu groß')
			return save, unsave
		global mögliche_bombenanordnungen
		mögliche_bombenanordnungen=[]
		
def helpme2():
	print('help')
	global field
	global open_fields
	global flagged_fields
	global save
	global unsave
	if (unsave).any() or (save).any():
		return save, unsave
	mx=field.shape[0]
	my=field.shape[1]
	verringertes_open_fields=np.zeros_like(open_fields)
	for pos in np.argwhere(open_fields==0):
		all_neighbors=pos+np.array([[0,1],[0,-1],[1,0],[-1,0],[1,-1],[-1,-1],[-1,1],[1,1]])
		yRand=np.isin(all_neighbors[:,1],[my,-1],invert=True)
		xRand=np.isin(all_neighbors[:,0],[mx,-1],invert=True)
		Rand=np.logical_and(xRand,yRand)
		all_neighbors=all_neighbors[Rand]
		for p in all_neighbors:
			verringertes_open_fields[p[0],p[1]]=1
	bRand=open_fields-verringertes_open_fields #bunter Rand ist wo array 0 ist

	verringertes_open_fields=np.zeros_like(open_fields)
	for pos in np.argwhere(open_fields==1):
		all_neighbors=pos+np.array([[0,1],[0,-1],[1,0],[-1,0],[1,-1],[-1,-1],[-1,1],[1,1]])
		yRand=np.isin(all_neighbors[:,1],[my,-1],invert=True)
		xRand=np.isin(all_neighbors[:,0],[mx,-1],invert=True)
		Rand=np.logical_and(xRand,yRand)
		all_neighbors=all_neighbors[Rand]
		for p in all_neighbors:
			verringertes_open_fields[p[0],p[1]]=1
	uRand=open_fields-verringertes_open_fields #verdeckter Rand ist wo array -1 ist
	print( np.argwhere(uRand==-1))
	quit()
	for pos in np.argwhere(bRand==0): #für jedes bunte feld am rand
		all_neighbors=pos+np.array([[0,1],[0,-1],[1,0],[-1,0],[1,-1],[-1,-1],[-1,1],[1,1]])
		yRand=np.isin(all_neighbors[:,1],[my,-1],invert=True)
		xRand=np.isin(all_neighbors[:,0],[mx,-1],invert=True)
		Rand=np.logical_and(xRand,yRand)
		all_neighbors=all_neighbors[Rand] #alle nachbarfelder ohne Spielfeldrand
		square=np.zeros((3,3))
		for p in all_neighbors: #p ist die echte position im spielfeld, sp ist die position im 3x3Feld
			sp=p-pos+np.array([1,1])
			if open_fields[p[0],p[1]]:
				square[sp[0],sp[1]]=0
			elif flagged_fields[p[0],p[1]]:
				square[sp[0],sp[1]]=-1
			else:
				square[sp[0],sp[1]]=1
		#print(square)
		#plt.imshow(square.T)
		#plt.show()
		kachelwert=field[pos[0],pos[1]]
		nnflags=len(np.argwhere(square.reshape(9)==-1))
		nclosed=len(np.argwhere(square.reshape(9)==1))
		#print(kachelwert,nnflags,nclosed)
		if kachelwert-nnflags==nclosed:
			unsavepos=np.argwhere(square==1)+pos-np.array([1,1])
			for pp in unsavepos:
				unsave[pp[0],pp[1]]=1
		if kachelwert==nnflags:
			savepos=np.argwhere(square==1)+pos-np.array([1,1])
			for pp in savepos:
				save[pp[0],pp[1]]=1
	if save.any() or unsave.any():
		return save, unsave
	else:
		global listuRand
		listuRand=np.argwhere(uRand+flagged_fields==-1)
		n=len(listuRand) # anzahl der unbekannten Randfelder
		I=2**n
		print('anzahl der unbekannten Randfelder:',n,'Anzahl der Bombenkombinationen',I)
		#print(uRand)
		#plt.imshow(uRand.T)
		#plt.show()
		if n>13:
			print('zu groß')
			return save, unsave
		global mögliche_bombenanordnungen
		mögliche_bombenanordnungen=[]
		if 0: #experimentell
			#divide urand in 8ter päckchen
			psize=8 #päckchengröße
			rn=n%psize
			nn=int((n-rn)/psize)
			print(nn,rn)
			for i1 in range(nn): #für jedes päcken
				mögliche_bombenanordnungen_päckchen=[]
				for i2 in range(2**psize):
					binär=np.binary_repr(i2)
					while len(binär)<psize:
						binär='0'+binär
					binär=[int(b) for b in binär]
					#print(binär)
					#füllt die vermuteten Minen in das uBombloc array
					uBombloc=np.zeros_like(uRand)
					for pos,bomb in zip(listuRand[i1*psize:(i1+1)*psize],binär): 
						uBombloc[pos[0],pos[1]]=bomb 
					#suche nach konflikten
					konflikt=False
					anzahl_bomben_in_nachbarschaft=0
					for pos in np.argwhere(bRand==0): #für jedes bunte feld am rand

						all_neighbors=pos+np.array([[0,1],[0,-1],[1,0],[-1,0],[1,-1],[-1,-1],[-1,1],[1,1]])
						yRand=np.isin(all_neighbors[:,1],[my,-1],invert=True)
						xRand=np.isin(all_neighbors[:,0],[mx,-1],invert=True)
						Rand=np.logical_and(xRand,yRand)
						all_neighbors=all_neighbors[Rand] #alle nachbarfelder ohne Spielfeldrand
						anzahl_bomben_in_nachbarschaft=0
						for p in all_neighbors: #p ist die echte position im spielfeld
							if flagged_fields[p[0],p[1]]:
								anzahl_bomben_in_nachbarschaft+=1
							elif uBombloc[p[0],p[1]]:
								anzahl_bomben_in_nachbarschaft+=1
						if not anzahl_bomben_in_nachbarschaft==field[pos[0],pos[1]]:
							konflikt=True
							break
					if not konflikt:
						#print('bunte feldnummer',field[pos[0],pos[1]],'anzahl_bomben_in_nachbarschaft', anzahl_bomben_in_nachbarschaft)
						#print('pos',pos)
						#print(field)
						mögliche_bombenanordnungen_päckchen.append(binär)
				mögliche_bombenanordnungen.append(mögliche_bombenanordnungen_päckchen)
			mögliche_bombenanordnungen_päckchen=[]
			for i2 in range(2**rn):
				binär=np.binary_repr(i2)
				while len(binär)<rn:
					binär='0'+binär
				binär=[int(b) for b in binär]
				print(binär,'#################')
				#füllt die vermuteten Minen in das uBombloc array
				uBombloc=np.zeros_like(uRand)
				for pos,bomb in zip(listuRand[-rn:],binär): 
					uBombloc[pos[0],pos[1]]=bomb 
					print('put bomb here',pos,bomb)
				#suche nach konflikten
				konflikt=False
				print(uBombloc)
				for pos in np.argwhere(bRand==0): #für jedes bunte feld am rand
					print('bfeld',pos)
					all_neighbors=pos+np.array([[0,1],[0,-1],[1,0],[-1,0],[1,-1],[-1,-1],[-1,1],[1,1]])
					yRand=np.isin(all_neighbors[:,1],[my,-1],invert=True)
					xRand=np.isin(all_neighbors[:,0],[mx,-1],invert=True)
					Rand=np.logical_and(xRand,yRand)
					all_neighbors=all_neighbors[Rand] #alle nachbarfelder ohne Spielfeldrand
					anzahl_bomben_in_nachbarschaft=0
					for p in all_neighbors: #p ist die echte position im spielfeld
						print(p)
						if flagged_fields[p[0],p[1]]:
							anzahl_bomben_in_nachbarschaft+=1
							print('flag here')
						elif uBombloc[p[0],p[1]]:
							anzahl_bomben_in_nachbarschaft+=1
							print('bomb here')
					if not anzahl_bomben_in_nachbarschaft==field[pos[0],pos[1]]:
						konflikt=True
						print('konflikt')
						break
				if not konflikt:
					#print('bunte feldnummer',field[pos[0],pos[1]],'anzahl_bomben_in_nachbarschaft', anzahl_bomben_in_nachbarschaft)
					#print('pos',pos)
					#print(field)
					mögliche_bombenanordnungen_päckchen.append(binär)
			mögliche_bombenanordnungen.append(mögliche_bombenanordnungen_päckchen)
			print(mögliche_bombenanordnungen)
			quit()
		ii=0
		for i in range(I):
			#erstelle den binärvector
			binär=np.binary_repr(i)
			while len(binär)<n:
				binär='0'+binär
			binär=[int(b) for b in binär]
			if np.sum(binär)<=(np.sum(bombloc)-np.sum(flagged_fields)): #wenn nur die maximale anzahl der vorhandenen Minen benutzt wird
				#füllt die vermuteten Minen in das uBombloc array
				ii+=1
				uBombloc=np.zeros_like(uRand)
				for pos,bomb in zip(listuRand,binär): 
					uBombloc[pos[0],pos[1]]=bomb 
				#suche nach konflikten
				konflikt=False
				anzahl_bomben_in_nachbarschaft=0
				for pos in np.argwhere(bRand==0): #für jedes bunte feld am rand

					all_neighbors=pos+np.array([[0,1],[0,-1],[1,0],[-1,0],[1,-1],[-1,-1],[-1,1],[1,1]])
					yRand=np.isin(all_neighbors[:,1],[my,-1],invert=True)
					xRand=np.isin(all_neighbors[:,0],[mx,-1],invert=True)
					Rand=np.logical_and(xRand,yRand)
					all_neighbors=all_neighbors[Rand] #alle nachbarfelder ohne Spielfeldrand
					anzahl_bomben_in_nachbarschaft=0
					for p in all_neighbors: #p ist die echte position im spielfeld
						if flagged_fields[p[0],p[1]]:
							anzahl_bomben_in_nachbarschaft+=1
						elif uBombloc[p[0],p[1]]:
							anzahl_bomben_in_nachbarschaft+=1
					if not anzahl_bomben_in_nachbarschaft==field[pos[0],pos[1]]:
						konflikt=True
						break
				if not konflikt:
					#print('bunte feldnummer',field[pos[0],pos[1]],'anzahl_bomben_in_nachbarschaft', anzahl_bomben_in_nachbarschaft)
					#print('pos',pos)
					#print(field)
					mögliche_bombenanordnungen.append(binär)
		print(ii,'mögliche_bombenanordnungen')
		mögliche_bombenanordnungen=np.array(mögliche_bombenanordnungen)
		for i in range(len(mögliche_bombenanordnungen[0,:])):
			if open_fields.any() and len(listuRand):
				pos=listuRand[i]
				if (mögliche_bombenanordnungen[:,i]==1).all():
					unsave[pos[0],pos[1]]=1
				if (mögliche_bombenanordnungen[:,i]==0).all():
					save[pos[0],pos[1]]=1
		return save, unsave

def AutoSolve():
	global automode
	print('automode',automode)
	global AutoSolveVar
	automode=AutoSolveVar.get()
	global open_fields
	global bombloc
	global field
	while automode:
		leftclick_Auto(open_fields,bombloc,field)
		root.update()
		#time.sleep(.5)

def HandleSettings():
	if DetectInconclusivityVar.get():
		ButtonGuess.configure(state='normal')
		EntryShowSolution.configure(state='normal')
		EntryAutoGuess.configure(state='normal')
		if ShowSolutionVar.get():
			EntryAutoSolve.configure(state='normal')
		else:
			EntryAutoSolve.configure(state='disabled')
	else:
		ShowSolutionVar.get()
		ButtonGuess.configure(state='disabled')
		EntryShowSolution.configure(state='disabled')
		EntryAutoGuess.configure(state='disabled')
		EntryAutoSolve.configure(state='disabled')
	spielfeld_anzeigen()

	
	



	
	
root = tk.Tk()
root.title('Minesweeper')
FrameMenu=tk.Frame(root)
FrameMenu.grid(row=0,column=0)
FrameMinefield=tk.Frame(root)
FrameMinefield.grid(row=1,column=0)

ButtonNewGame = tk.Button(FrameMenu, text='New Game',command=newgame)
ButtonNewGame.grid(row=0,column=0)
LabelNx =tk.Label(FrameMenu,text='Breite:')
LabelNx.grid(row=1,column=0)
EntryNx=tk.Entry(FrameMenu)
EntryNx.grid(row=1,column=1)
EntryNx.insert(tk.END, str(defaultx))
LabelNy =tk.Label(FrameMenu,text='Höhe:')
LabelNy.grid(row=2,column=0)
EntryNy=tk.Entry(FrameMenu)
EntryNy.grid(row=2,column=1)
EntryNy.insert(tk.END, str(defaulty))
LabelNm =tk.Label(FrameMenu,text='Minen:')
LabelNm.grid(row=3,column=0)
EntryNm=tk.Entry(FrameMenu)
EntryNm.grid(row=3,column=1)
EntryNm.insert(tk.END, str(defaultm))
AusblendenVar = tk.IntVar()
EntryAusblenden=tk.Checkbutton(FrameMenu, text='Ausblenden',variable=AusblendenVar, command=spielfeld_anzeigen)
EntryAusblenden.grid(sticky="W", row=4,column=1)
ShowSolutionVar = tk.IntVar()
EntryShowSolution = tk.Checkbutton(FrameMenu, text='Show Solution',variable=ShowSolutionVar,command=HandleSettings)
EntryShowSolution.grid(sticky="W", row=3,column=4)
DetectInconclusivityVar=tk.IntVar()
EntryDetectInconclusivity = tk.Checkbutton(FrameMenu, text='Detect Inconclusivity',variable=DetectInconclusivityVar, command=HandleSettings)
EntryDetectInconclusivity.grid(sticky="W", row=0,column=4)
AutoGuessVar=tk.IntVar()
EntryAutoGuess = tk.Checkbutton(FrameMenu, text='Auto Guess',variable=AutoGuessVar)
EntryAutoGuess.grid(sticky="W", row=2,column=4)
AutoSolveVar=tk.IntVar()
EntryAutoSolve = tk.Checkbutton(FrameMenu, text='Auto Solve',variable=AutoSolveVar,command=AutoSolve)
EntryAutoSolve.grid(sticky="W", row=4,column=4)
ButtonGuess = tk.Button(FrameMenu, text='Guess for me',command=Guess,state='disabled')
ButtonGuess.grid(sticky="W", row=1,column=4)
LabelRestminen =tk.Label(FrameMenu,text='999')
LabelRestminen.grid(row=0,column=1)





button = tk.Button(root, text='Stop', width=25, command=root.destroy)
button.grid(row=2,column=0)


if automode: #generate train_data
	max_fields_opend=np.random.randint(500)
	columns=[]
	for p in np.argwhere(np.ones((defaultx,defaulty))):
		for i in ['poi','open','value']:
			columns.append('('+str(p[0])+','+str(p[1])+') '+i)
	columns+=['is_bomb']
	#print(columns)
			

	df_learn=pd.DataFrame([],columns=columns)#['xpos','ypos']+[ '('+str(i[0])+','+str(i[1])+')' for i in np.argwhere(np.ones((defaultx,defaulty)))]+['is_bomb'])
	print(df_learn)
	im=newgame()
	ShowSolutionVar.set(1)
	AutoSolve()



else:
	im=newgame()


root.mainloop()