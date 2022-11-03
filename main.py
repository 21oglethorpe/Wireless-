from tkinter import *
with open('map.txt', 'r') as f:
    l = [[int(num) for num in line.split(',')] for line in f]
root = Tk()
root.title('tester')
app_width = 750
app_height = 750
root.geometry(f'{app_width}x{app_height}')
sim = {
    0: ' ',
    1: '-',
    2: '|'
}
for rows in l:
    concat = ""
    lister = []
    for element in rows:
        lister.append(sim[element])
        concat = concat + " " + sim[element]
    my_label = Label(root, text = concat, font=("Arial", 10))
    my_label.pack(pady = .05)
root.mainloop()
