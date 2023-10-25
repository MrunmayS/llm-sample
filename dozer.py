from pydozer.api import ApiClient
from pydozer.ingest import IngestClient

api_client = ApiClient("credit_endpt")
customer_client = ApiClient("financial_profile")
def getCredit(id):
    id+=23200
    data = api_client.query({"$filter" : {"id":id}})
    rec = data.records[0]
    return rec.record.values[1].int_value

def getCustomerData(input):
    data = customer_client.query({"$filter": {"name":input}})
    rec = data.records[0].record
    name = rec.values[0].string_value
    income = rec.values[1].int_value
    age = rec.values[2].int_value
    dependents = rec.values[3].int_value
    credit_amt = rec.values[4].int_value
    repay_status = rec.values[5].float_value
    util_ratio = rec.values[6].float_value
    address = rec.values[7].string_value

    return [name,income,age,dependents,credit_amt,repay_status,util_ratio,address]


# data = customer_client.query({"$filter": {"name": "Lee Ryan"} })
# print(data.records[0].record)
