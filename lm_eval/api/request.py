from typing import Any, Optional


REQUEST_RETURN_LENGTHS = {
    "loglikelihood": 2,
    "greedy_until": None,
    "loglikelihood_rolling": None,
}

class Request:
    def __init__(
        self, 
        request_type: str,
        args: Optional[Any] = None,
        index: Optional[int] = None,
        unique_request_id: int = None,
        doc_id: int = None
    ):
        if request_type not in REQUEST_RETURN_LENGTHS.keys():
            raise NotImplementedError(
                "The request type {} is not implemented!".format(request_type)
            )
        self.request_type = request_type
        self.args = args
        self.index = index
        self.unique_request_id = unique_request_id
        self.doc_id = doc_id

    def __iter__(self):
        if REQUEST_RETURN_LENGTHS[self.request_type] is None:
            raise IndexError("This request type does not return multiple arguments!")
        for return_index in range(REQUEST_RETURN_LENGTHS[self.request_type]):
            yield Request(self.request_type, self.args, return_index, self.unique_request_id, self.doc_id)

    def __getitem__(self, i: int):
        if REQUEST_RETURN_LENGTHS[self.request_type] is None:
            raise IndexError("This request type does not return multiple arguments!")
        return Request(self.request_type, self.args, i, self.unique_request_id, self.doc_id)

    def __eq__(self, other: "Request"):
        return (
            self.request_type == other.request_type
            and self.args == other.args
            and self.index == other.index
            and self.unique_request_id == other.unique_request_id
            and self.doc_id == other.doc_id
        )

    def __repr__(self):
        return f"Req_{self.request_type}{self.args}[{self.index}]_{self.doc_id}\n"


class RequestFactory:
    def __getattr__(self, attr):
        def fn(*args):
            return Request(attr, args)

        return fn


rf = RequestFactory()
