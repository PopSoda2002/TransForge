from cs336_basics.utils import DoublyLinkedList

def test_double_linked_list():
    dll = DoublyLinkedList()
    dll.add_node(b"a")
    dll.add_node(b"b")
    dll.add_node(b"c")
    assert dll.head.next.val == b"a"
    assert dll.tail.prev.val == b"c"
    assert dll.head.next.next.val == b"b"
    assert dll.tail.prev.prev.val == b"b"

def test_double_linked_list_merge_pair():
    dll = DoublyLinkedList()
    dll.add_node(b"a")
    dll.add_node(b"b")
    dll.add_node(b"c")
    dll.merge_pair_and_get_deltas((b"a", b"b"))
    assert dll.head.next.val == b"ab"
    assert dll.tail.prev.val == b"c"
    assert dll.head.next.next.val == b"c"
    assert dll.tail.prev.prev.val == b"ab"
