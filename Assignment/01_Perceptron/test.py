def list_of_list_to_csv(list):
	csv = open("output_data.csv", "w")
	for inner_list in list:
		string = ','.join(map(str, inner_list)) + '\n'
		csv.write(string)
	csv.close()

def csv_to_list_of_list():
	csv = open("output_data.csv", "a")
	csv.close()
	csv = open("output_data.csv", "r")
	list = []
	for line in csv:
		inner_list = line.strip().split(',')
		list.append(inner_list)
	return list