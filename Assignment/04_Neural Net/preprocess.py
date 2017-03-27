file_in = open("ps5_data-labels.csv", "r")
file_out = open("ps5_data-labels_edit.csv", "w")

for line in file_in:
	file_out.write(str(int(line.strip())-1)+"\n")

file_out.close()
file_in.close()